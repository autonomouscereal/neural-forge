"""Atomic adaptive surgery for coupled Qwen3 GQA group paths.

The ordinary Qwen attention codebook stores a rectangular number of depth
prototypes for every canonical KV group.  Removing one prototype from only one
group therefore cannot reclaim physical storage.  This bridge converts an
accepted proposal to a global, ragged prototype pool: every pool entry owns a
complete Q/K/V/O group bundle and every logical ``[layer, group]`` execution
selects one entry.  Nonadjacent and cross-group redirects can then remove one
real bundle immediately without deleting any logical attention path.

No partial projection edit is representable here.  A tentative registry is
built separately and all Q, K, V, and O views are installed as one transaction.
The bridge never trains, never calls backward, and never interprets a gate
acceptance as evidence that model quality improved.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import operator
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from neural_forge.adaptive_surgery_graph import (
    AdaptiveSurgeryGraph,
    AnchorSpec,
    EvaluationDecision,
    MergeCandidate,
    ScoreWeights,
    SurgeryNode,
    SurgerySearchExhausted,
    SurgerySnapshot,
    SurgeryTransaction,
)
from neural_forge.hf_attention_codebook import (
    AttentionCodebookRegistry,
    GlobalAttentionCodebook,
    RaggedAttentionCodebookRegistry,
)
from neural_forge.hf_qwen_codebook_model import (
    Qwen3CodebookError,
    Qwen3CodebookParameterReport,
    Qwen3CodebookStateError,
    qwen3_codebook_parameter_report,
)


SCHEMA_VERSION = 1
QWEN3_ATTENTION_NODE_KIND = "gqa-group"
_QWEN3_REPORT_ATTRIBUTE = "neural_forge_codebook_report"
Qwen3AttentionNodeId = tuple[str, str, int, int]
Gate = Callable[..., bool | EvaluationDecision | Mapping[str, Any]]
FeatureRefresh = Callable[..., Any]
CandidateSelector = Callable[..., Any]


class Qwen3AttentionSurgeryError(Qwen3CodebookError):
    """Base error for Qwen3 attention-surgery validation and execution."""


class Qwen3AttentionSurgeryShapeError(Qwen3AttentionSurgeryError):
    """A feature, projection, candidate, or tensor has an invalid shape."""


class Qwen3AttentionSurgeryStateError(
    Qwen3AttentionSurgeryError, Qwen3CodebookStateError
):
    """A prepared runtime cannot support an exact attention transaction."""


class Qwen3AttentionSurgeryTargetError(Qwen3AttentionSurgeryError):
    """A requested physical attention-registry byte target was not reached."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise Qwen3AttentionSurgeryShapeError(f"{name} must be a non-negative integer.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise Qwen3AttentionSurgeryShapeError(
            f"{name} must be a non-negative integer."
        ) from error
    if result < 0:
        raise Qwen3AttentionSurgeryShapeError(f"{name} must be a non-negative integer.")
    return result


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result <= 0:
        raise Qwen3AttentionSurgeryShapeError(f"{name} must be positive.")
    return result


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Qwen3AttentionSurgeryShapeError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise Qwen3AttentionSurgeryShapeError(
            f"{name} must be a finite number."
        ) from error
    if not math.isfinite(result):
        raise Qwen3AttentionSurgeryShapeError(f"{name} must be finite.")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise Qwen3AttentionSurgeryShapeError(f"{name} must be positive.")
    return result


def _optional_nonnegative_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    result = _finite_float(value, name)
    if result < 0.0:
        raise Qwen3AttentionSurgeryShapeError(f"{name} cannot be negative.")
    return result


def _feature_vector(value: Any, name: str) -> tuple[float, ...]:
    if value is None:
        raise Qwen3AttentionSurgeryShapeError(f"{name} is required.")
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        numpy_method = getattr(value, "numpy", None)
        if callable(numpy_method):
            value = numpy_method()
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise Qwen3AttentionSurgeryShapeError(
            f"{name} must be a numeric vector."
        ) from error
    if array.ndim == 0 or array.size == 0:
        raise Qwen3AttentionSurgeryShapeError(
            f"{name} must contain at least one value."
        )
    flattened = np.ascontiguousarray(array.reshape(-1), dtype=np.float64)
    if not bool(np.isfinite(flattened).all()):
        raise Qwen3AttentionSurgeryShapeError(
            f"{name} must contain only finite values."
        )
    return tuple(float(item) for item in flattened)


def _optional_feature_vector(value: Any, name: str) -> tuple[float, ...] | None:
    return None if value is None else _feature_vector(value, name)


def _neutral_feature_vector(value: Any, name: str) -> tuple[float, ...]:
    return (0.0,) if value is None else _feature_vector(value, name)


def _coalesce(primary: Any, alias: Any, primary_name: str, alias_name: str) -> Any:
    if primary is not None and alias is not None:
        raise Qwen3AttentionSurgeryShapeError(
            f"Provide only one of {primary_name} and {alias_name}."
        )
    return primary if primary is not None else alias


def _anchor(value: AnchorSpec | str | None) -> AnchorSpec | None:
    if value is None or isinstance(value, AnchorSpec):
        return value
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"exact", "pinned", "hard"}:
        return AnchorSpec.pinned()
    if normalized in {"projected", "projection"}:
        return AnchorSpec.projected()
    if normalized == "soft":
        return AnchorSpec.soft()
    raise Qwen3AttentionSurgeryShapeError(f"Unknown anchor mode {value!r}.")


def qwen3_attention_group_node_id(
    layer_index: Any, group_index: Any
) -> Qwen3AttentionNodeId:
    """Return the stable ID for one real coupled Q/K/V/O source group."""

    return (
        "qwen3",
        QWEN3_ATTENTION_NODE_KIND,
        _nonnegative_int(layer_index, "layer_index"),
        _nonnegative_int(group_index, "group_index"),
    )


def _node_ref(value: Any, name: str = "node") -> Qwen3AttentionNodeId:
    if isinstance(value, Qwen3AttentionSurgeryFeature):
        return value.node_id
    if isinstance(value, tuple) and len(value) == 2:
        return qwen3_attention_group_node_id(value[0], value[1])
    if (
        isinstance(value, tuple)
        and len(value) == 4
        and value[:2] == ("qwen3", QWEN3_ATTENTION_NODE_KIND)
    ):
        return qwen3_attention_group_node_id(value[2], value[3])
    raise Qwen3AttentionSurgeryShapeError(
        f"{name} must be (layer, group) or a canonical Qwen3 attention node ID."
    )


@dataclass(frozen=True, slots=True, init=False)
class Qwen3AttentionSurgeryFeature:
    """Bounded activation, rollout/JVP, and output evidence for one GQA group."""

    layer_index: int
    group_index: int
    weight_features: tuple[float, ...] | None
    activation_features: tuple[float, ...]
    jvp_features: tuple[float, ...]
    output_features: tuple[float, ...]
    domain_features: tuple[float, ...] | None
    importance: float
    anchor: AnchorSpec | None

    def __init__(
        self,
        layer_index: Any,
        group_index: Any = None,
        weight: Any = None,
        activation: Any = None,
        jvp: Any = None,
        output: Any = None,
        importance: float = 1.0,
        anchor: AnchorSpec | str | None = None,
        *,
        kv_group_index: Any = None,
        weight_features: Any = None,
        activation_features: Any = None,
        rollout_jvp_features: Any = None,
        output_dual_features: Any = None,
        domain_features: Any = None,
    ) -> None:
        group = _coalesce(group_index, kv_group_index, "group_index", "kv_group_index")
        if group is None:
            raise Qwen3AttentionSurgeryShapeError(
                "group_index or kv_group_index is required."
            )
        weight_value = _coalesce(weight, weight_features, "weight", "weight_features")
        activation_value = _coalesce(
            activation,
            activation_features,
            "activation",
            "activation_features",
        )
        jvp_value = _coalesce(jvp, rollout_jvp_features, "jvp", "rollout_jvp_features")
        output_value = _coalesce(
            output,
            output_dual_features,
            "output",
            "output_dual_features",
        )
        object.__setattr__(
            self, "layer_index", _nonnegative_int(layer_index, "layer_index")
        )
        object.__setattr__(self, "group_index", _nonnegative_int(group, "group_index"))
        object.__setattr__(
            self,
            "weight_features",
            _optional_feature_vector(weight_value, "weight features"),
        )
        object.__setattr__(
            self,
            "activation_features",
            _feature_vector(activation_value, "activation features"),
        )
        object.__setattr__(
            self, "jvp_features", _neutral_feature_vector(jvp_value, "JVP features")
        )
        object.__setattr__(
            self,
            "output_features",
            _neutral_feature_vector(output_value, "output/dual features"),
        )
        object.__setattr__(
            self,
            "domain_features",
            _optional_feature_vector(domain_features, "domain features"),
        )
        object.__setattr__(
            self, "importance", _positive_float(importance, "importance")
        )
        object.__setattr__(self, "anchor", _anchor(anchor))

    @property
    def node_id(self) -> Qwen3AttentionNodeId:
        return qwen3_attention_group_node_id(self.layer_index, self.group_index)

    @property
    def rollout_jvp_features(self) -> tuple[float, ...]:
        return self.jvp_features

    @property
    def output_dual_features(self) -> tuple[float, ...]:
        return self.output_features


Qwen3GQASurgeryFeature = Qwen3AttentionSurgeryFeature


@dataclass(frozen=True, slots=True, init=False)
class Qwen3AttentionProjection:
    """Executable role scales plus evidence about the space used to fit them.

    Q/K scalars are never described as preserving Qwen's normalized query/key
    geometry.  Automatic fits keep Q/K at one and score affine directions only
    after the corresponding q_norm/k_norm weights are applied.
    """

    source: Qwen3AttentionNodeId
    target: Qwen3AttentionNodeId
    q: float
    k: float
    v: float
    o: float
    activation_residual: float | None
    jvp_residual: float | None
    qk_normalized_residual: float | None
    bias_included: bool
    fit_space: str

    def __init__(
        self,
        source: Any,
        target: Any,
        q: Any,
        k: Any,
        v: Any,
        o: Any,
        *,
        activation_residual: Any = None,
        jvp_residual: Any = None,
        qk_normalized_residual: Any = None,
        bias_included: bool = False,
        fit_space: str = "explicit-unverified",
    ) -> None:
        source_id = _node_ref(source, "projection source")
        target_id = _node_ref(target, "projection target")
        if source_id == target_id:
            raise Qwen3AttentionSurgeryShapeError(
                "Projection source and target must be distinct group paths."
            )
        object.__setattr__(self, "source", source_id)
        object.__setattr__(self, "target", target_id)
        for role, value in (("q", q), ("k", k), ("v", v), ("o", o)):
            if value is None:
                raise Qwen3AttentionSurgeryShapeError(
                    "A coupled attention projection requires q, k, v, and o; "
                    f"{role} is missing."
                )
            object.__setattr__(
                self, role, _finite_float(value, f"{role} projection coefficient")
            )
        object.__setattr__(
            self,
            "activation_residual",
            _optional_nonnegative_float(
                activation_residual, "activation projection residual"
            ),
        )
        object.__setattr__(
            self,
            "jvp_residual",
            _optional_nonnegative_float(jvp_residual, "JVP projection residual"),
        )
        object.__setattr__(
            self,
            "qk_normalized_residual",
            _optional_nonnegative_float(
                qk_normalized_residual, "normalized Q/K projection residual"
            ),
        )
        if not isinstance(bias_included, bool):
            raise Qwen3AttentionSurgeryShapeError("bias_included must be boolean.")
        if not isinstance(fit_space, str) or not fit_space.strip():
            raise Qwen3AttentionSurgeryShapeError("fit_space must be a non-empty string.")
        object.__setattr__(self, "bias_included", bias_included)
        object.__setattr__(self, "fit_space", fit_space.strip())

    @property
    def values(self) -> tuple[float, float, float, float]:
        return self.q, self.k, self.v, self.o

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": list(self.source),
            "target": list(self.target),
            "q": self.q,
            "k": self.k,
            "v": self.v,
            "o": self.o,
            "activation_residual": self.activation_residual,
            "jvp_residual": self.jvp_residual,
            "qk_normalized_residual": self.qk_normalized_residual,
            "bias_included": self.bias_included,
            "fit_space": self.fit_space,
        }


Qwen3GQAProjection = Qwen3AttentionProjection


def _feature_sequence(
    values: Iterable[Any],
) -> tuple[Qwen3AttentionSurgeryFeature, ...]:
    if isinstance(values, (str, bytes)):
        raise Qwen3AttentionSurgeryShapeError("features must be an iterable.")
    try:
        result = tuple(values)
    except TypeError as error:
        raise Qwen3AttentionSurgeryShapeError(
            "features must be an iterable."
        ) from error
    if not result or not all(
        isinstance(item, Qwen3AttentionSurgeryFeature) for item in result
    ):
        raise Qwen3AttentionSurgeryShapeError(
            "Every feature must be a Qwen3AttentionSurgeryFeature."
        )
    return result


def _projection_sequence(
    values: Sequence[Qwen3AttentionProjection] | Mapping[Any, Any],
) -> tuple[Qwen3AttentionProjection, ...]:
    result: list[Qwen3AttentionProjection] = []
    if isinstance(values, Mapping):
        for pair, payload in values.items():
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise Qwen3AttentionSurgeryShapeError(
                    "Projection mapping keys must be (source, target) pairs."
                )
            if isinstance(payload, Qwen3AttentionProjection):
                item = payload
                if (item.source, item.target) != (
                    _node_ref(pair[0]),
                    _node_ref(pair[1]),
                ):
                    raise Qwen3AttentionSurgeryShapeError(
                        "Projection mapping key does not match its value."
                    )
            elif isinstance(payload, Mapping):
                missing = [role for role in ("q", "k", "v", "o") if role not in payload]
                if missing:
                    raise Qwen3AttentionSurgeryShapeError(
                        "Partial Q/K/V/O projection edits are forbidden; missing "
                        f"{missing!r}."
                    )
                item = Qwen3AttentionProjection(
                    pair[0],
                    pair[1],
                    payload["q"],
                    payload["k"],
                    payload["v"],
                    payload["o"],
                    activation_residual=payload.get("activation_residual"),
                    jvp_residual=payload.get("jvp_residual"),
                )
            else:
                try:
                    q, k, v, o = payload
                except (TypeError, ValueError) as error:
                    raise Qwen3AttentionSurgeryShapeError(
                        "Projection values must contain all q, k, v, and o scalars."
                    ) from error
                item = Qwen3AttentionProjection(pair[0], pair[1], q, k, v, o)
            result.append(item)
    else:
        if isinstance(values, (str, bytes)):
            raise Qwen3AttentionSurgeryShapeError(
                "projections must be a sequence or mapping."
            )
        try:
            entries = tuple(values)
        except TypeError as error:
            raise Qwen3AttentionSurgeryShapeError(
                "projections must be a sequence or mapping."
            ) from error
        if not all(isinstance(item, Qwen3AttentionProjection) for item in entries):
            raise Qwen3AttentionSurgeryShapeError(
                "Every projection must be a Qwen3AttentionProjection."
            )
        result.extend(entries)
    by_pair: dict[
        tuple[Qwen3AttentionNodeId, Qwen3AttentionNodeId],
        Qwen3AttentionProjection,
    ] = {}
    for item in result:
        pair = (item.source, item.target)
        if pair in by_pair:
            raise Qwen3AttentionSurgeryShapeError(
                f"Duplicate attention projection for {pair!r}."
            )
        by_pair[pair] = item
    return tuple(by_pair[pair] for pair in sorted(by_pair))


@dataclass(frozen=True, slots=True, init=False)
class Qwen3AttentionEvidenceRefresh:
    """Accepted-only replacement evidence for the remaining real prototypes."""

    features: tuple[Qwen3AttentionSurgeryFeature, ...]
    projections: tuple[Qwen3AttentionProjection, ...]

    def __init__(
        self,
        features: Iterable[Qwen3AttentionSurgeryFeature],
        projections: (Sequence[Qwen3AttentionProjection] | Mapping[Any, Any]) = (),
    ) -> None:
        object.__setattr__(self, "features", _feature_sequence(features))
        object.__setattr__(self, "projections", _projection_sequence(projections))


@dataclass(frozen=True, slots=True)
class Qwen3AttentionFeatureBounds:
    max_nodes: int = 1_000_000
    max_weight_dimensions: int = 256
    max_activation_dimensions: int = 4_096
    max_jvp_dimensions: int = 4_096
    max_output_dimensions: int = 4_096
    max_domain_dimensions: int = 4_096
    max_total_values: int = 64_000_000
    derived_weight_dimensions: int = 64

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))


def _tensor_sha256(value: Tensor) -> str:
    detached = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "dtype": str(detached.dtype),
                "shape": [int(item) for item in detached.shape],
            }
        )
    )
    bytes_view = detached.reshape(-1).view(torch.uint8)
    for start in range(0, int(bytes_view.numel()), 4 * 1024 * 1024):
        digest.update(bytes_view[start : start + 4 * 1024 * 1024].numpy().tobytes())
    return digest.hexdigest()


def _tensor_identity_manifest(module: nn.Module) -> tuple[tuple[Any, ...], ...]:
    entries: list[tuple[Any, ...]] = []
    for kind, values in (
        ("parameter", module.named_parameters(remove_duplicate=False)),
        ("buffer", module.named_buffers(remove_duplicate=False)),
    ):
        for name, value in values:
            try:
                version = int(value._version)
            except RuntimeError:
                version = -1
            entries.append(
                (
                    kind,
                    name,
                    id(value),
                    int(value.data_ptr()) if value.device.type != "meta" else 0,
                    version,
                    tuple(int(item) for item in value.shape),
                    str(value.dtype),
                    str(value.device),
                )
            )
    return tuple(entries)


def _assert_callback_tensor_contract(
    runtime: nn.Module,
    before: tuple[tuple[Any, ...], ...],
    callback_name: str,
) -> None:
    after = _tensor_identity_manifest(runtime)
    if after != before:
        raise Qwen3AttentionSurgeryStateError(
            f"Fatal out-of-contract mutation by {callback_name}; only the owned "
            "attention-codebook installation is rollback-capable, so no whole-model "
            "rollback claim is made."
        )


def _tensor_bytes(value: Tensor | None) -> int:
    return 0 if value is None else int(value.numel()) * int(value.element_size())


def _is_ragged(registry: AttentionCodebookRegistry) -> bool:
    return registry.prototype_q_weight.ndim == 3


def _prototype_count(registry: AttentionCodebookRegistry) -> int:
    if _is_ragged(registry):
        return int(registry.prototype_q_weight.shape[0])
    return int(
        registry.prototype_q_weight.shape[0] * registry.prototype_q_weight.shape[1]
    )


def _global_assignments(registry: AttentionCodebookRegistry) -> Tensor:
    assignments = registry.assignments.detach().to(device="cpu", dtype=torch.int64)
    if _is_ragged(registry):
        return assignments
    groups, slots = map(int, registry.prototype_q_weight.shape[:2])
    if int(assignments.shape[1]) != groups:
        raise Qwen3AttentionSurgeryStateError(
            "Attention assignments disagree with the prototype group axis."
        )
    offsets = torch.arange(groups, dtype=torch.int64) * slots
    return assignments + offsets.unsqueeze(0)


def _source_indices(registry: AttentionCodebookRegistry) -> Tensor:
    return (
        registry.prototype_source_indices.detach()
        .to(device="cpu", dtype=torch.int64)
        .reshape(-1)
    )


def _flat_prototype_tensor(
    registry: AttentionCodebookRegistry, name: str
) -> Tensor | None:
    value = getattr(registry, name, None)
    if value is None:
        return None
    if _is_ragged(registry):
        return value
    return value.reshape(-1, *value.shape[2:])


def _prototype_bundle_tensors(
    registry: AttentionCodebookRegistry, slot: int
) -> tuple[Tensor, ...]:
    result: list[Tensor] = []
    for name in (
        "prototype_q_weight",
        "prototype_k_weight",
        "prototype_v_weight",
        "prototype_o_weight",
        "prototype_q_bias",
        "prototype_k_bias",
        "prototype_v_bias",
    ):
        value = _flat_prototype_tensor(registry, name)
        if value is not None:
            result.append(value[slot])
    return tuple(result)


def _prototype_bundle_sha256(registry: AttentionCodebookRegistry, slot: int) -> str:
    return _sha256(
        [_tensor_sha256(value) for value in _prototype_bundle_tensors(registry, slot)]
    )


def _prototype_bytes(registry: AttentionCodebookRegistry) -> int:
    if _prototype_count(registry) <= 0:
        raise Qwen3AttentionSurgeryStateError("Attention registry has no prototypes.")
    result = sum(
        _tensor_bytes(value[0])
        for name in (
            "prototype_q_weight",
            "prototype_k_weight",
            "prototype_v_weight",
            "prototype_o_weight",
            "prototype_q_bias",
            "prototype_k_bias",
            "prototype_v_bias",
        )
        if (value := _flat_prototype_tensor(registry, name)) is not None
    )
    result += int(registry.prototype_source_indices.element_size())
    return result


@dataclass(frozen=True, slots=True)
class Qwen3AttentionPrototypeCoordinate:
    """Unambiguous canonical and physical provenance for one active slot."""

    active_slot: int
    source_layer: int
    canonical_group: int
    physical_group: int
    source_flat_index: int

    def as_dict(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name)) for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class Qwen3AttentionRuntimeSnapshot:
    logical_layers: int
    key_value_heads: int
    prototype_count: int
    assignments: tuple[tuple[int, ...], ...]
    prototype_source_indices: tuple[int, ...]
    prototype_coordinates: tuple[Qwen3AttentionPrototypeCoordinate, ...]
    assignment_source_indices: tuple[tuple[int, ...], ...]
    prototype_bundle_sha256s: tuple[tuple[Qwen3AttentionNodeId, str], ...]
    role_scales_present: bool
    tensor_bytes: int
    prototype_pool_bytes: int
    assignment_bytes: int
    scale_bytes: int
    index_bytes: int
    registry_sha256: str

    @property
    def active_node_ids(self) -> tuple[Qwen3AttentionNodeId, ...]:
        groups = self.key_value_heads
        return tuple(
            qwen3_attention_group_node_id(*divmod(source, groups))
            for source in self.prototype_source_indices
        )

    @property
    def deployable_attention_codebook_bytes(self) -> int:
        return self.tensor_bytes

    def bundle_hash(self, node: Any) -> str:
        node_id = _node_ref(node)
        try:
            return dict(self.prototype_bundle_sha256s)[node_id]
        except KeyError as error:
            raise Qwen3AttentionSurgeryStateError(
                f"{node_id!r} is not an active physical prototype."
            ) from error

    def as_dict(self) -> dict[str, Any]:
        return {
            "logical_layers": self.logical_layers,
            "key_value_heads": self.key_value_heads,
            "prototype_count": self.prototype_count,
            "assignments": [list(row) for row in self.assignments],
            "prototype_source_indices": list(self.prototype_source_indices),
            "prototype_coordinates": [
                coordinate.as_dict() for coordinate in self.prototype_coordinates
            ],
            "assignment_source_indices": [
                list(row) for row in self.assignment_source_indices
            ],
            "prototype_bundle_sha256s": [
                [list(node), digest] for node, digest in self.prototype_bundle_sha256s
            ],
            "role_scales_present": self.role_scales_present,
            "tensor_bytes": self.tensor_bytes,
            "prototype_pool_bytes": self.prototype_pool_bytes,
            "assignment_bytes": self.assignment_bytes,
            "scale_bytes": self.scale_bytes,
            "index_bytes": self.index_bytes,
            "registry_sha256": self.registry_sha256,
        }


def _registry_snapshot(
    registry: AttentionCodebookRegistry,
) -> Qwen3AttentionRuntimeSnapshot:
    states = registry.state_dict()
    state_hashes = {
        name: _tensor_sha256(value) for name, value in sorted(states.items())
    }
    tensor_bytes = sum(_tensor_bytes(value) for value in states.values())
    assignments_tensor = _global_assignments(registry)
    assignments = tuple(
        tuple(int(item) for item in row) for row in assignments_tensor.tolist()
    )
    groups = int(assignments_tensor.shape[1])
    sources = tuple(int(item) for item in _source_indices(registry).tolist())
    permutations = registry.group_permutations.detach().to(
        device="cpu", dtype=torch.int64
    )
    coordinates: list[Qwen3AttentionPrototypeCoordinate] = []
    for slot, source in enumerate(sources):
        source_layer, physical_group = divmod(source, groups)
        canonical_matches = torch.nonzero(
            permutations[source_layer] == physical_group,
            as_tuple=False,
        ).reshape(-1)
        if int(canonical_matches.numel()) != 1:
            raise Qwen3AttentionSurgeryStateError(
                "Group permutation lacks one canonical coordinate for provenance."
            )
        coordinates.append(
            Qwen3AttentionPrototypeCoordinate(
                active_slot=slot,
                source_layer=source_layer,
                canonical_group=int(canonical_matches.item()),
                physical_group=physical_group,
                source_flat_index=source,
            )
        )
    assignment_sources = tuple(
        tuple(sources[slot] for slot in row) for row in assignments
    )
    bundle_hashes = tuple(
        (
            qwen3_attention_group_node_id(*divmod(source, groups)),
            _prototype_bundle_sha256(registry, slot),
        )
        for slot, source in enumerate(sources)
    )
    prototype_names = {
        "prototype_q_weight",
        "prototype_k_weight",
        "prototype_v_weight",
        "prototype_o_weight",
        "prototype_q_bias",
        "prototype_k_bias",
        "prototype_v_bias",
        "prototype_source_indices",
    }
    prototype_pool_bytes = sum(
        _tensor_bytes(value)
        for name, value in states.items()
        if name in prototype_names
    )
    assignment_bytes = _tensor_bytes(registry.assignments)
    scale_bytes = _tensor_bytes(registry.role_scales)
    index_bytes = (
        assignment_bytes
        + _tensor_bytes(registry.prototype_source_indices)
        + _tensor_bytes(registry.group_permutations)
        + _tensor_bytes(getattr(registry, "_registry_schema_version", None))
        + _tensor_bytes(getattr(registry, "_registry_layout", None))
    )
    payload = {
        "logical_layers": int(assignments_tensor.shape[0]),
        "groups": groups,
        "assignments": assignments,
        "sources": sources,
        "prototype_coordinates": [item.as_dict() for item in coordinates],
        "state_hashes": state_hashes,
        "tensor_bytes": tensor_bytes,
    }
    return Qwen3AttentionRuntimeSnapshot(
        logical_layers=int(assignments_tensor.shape[0]),
        key_value_heads=groups,
        prototype_count=len(sources),
        assignments=assignments,
        prototype_source_indices=sources,
        prototype_coordinates=tuple(coordinates),
        assignment_source_indices=assignment_sources,
        prototype_bundle_sha256s=bundle_hashes,
        role_scales_present=registry.role_scales is not None,
        tensor_bytes=tensor_bytes,
        prototype_pool_bytes=prototype_pool_bytes,
        assignment_bytes=assignment_bytes,
        scale_bytes=scale_bytes,
        index_bytes=index_bytes,
        registry_sha256=_sha256(payload),
    )


def _validated_runtime(
    runtime: nn.Module,
) -> tuple[GlobalAttentionCodebook, AttentionCodebookRegistry]:
    if not isinstance(runtime, nn.Module):
        raise Qwen3AttentionSurgeryShapeError("runtime must be a torch nn.Module.")
    try:
        report = qwen3_codebook_parameter_report(runtime)
    except Qwen3CodebookError as error:
        raise Qwen3AttentionSurgeryStateError(
            "runtime must be prepared by prepare_qwen3_codebook_model()."
        ) from error
    if getattr(getattr(runtime, "config", None), "model_type", None) != "qwen3":
        raise Qwen3AttentionSurgeryStateError(
            "Attention surgery only supports a prepared Qwen3 runtime."
        )
    if not report.full_logical_depth:
        raise Qwen3AttentionSurgeryStateError(
            "Prepared Qwen3 runtime must retain full logical depth."
        )
    codebook = getattr(runtime, "attention_codebook", None)
    if not isinstance(codebook, GlobalAttentionCodebook):
        raise Qwen3AttentionSurgeryStateError(
            "runtime.attention_codebook must be a GlobalAttentionCodebook."
        )
    registry = codebook.registry
    if not isinstance(registry, AttentionCodebookRegistry):
        raise Qwen3AttentionSurgeryStateError(
            "Qwen3 attention codebook has no AttentionCodebookRegistry."
        )
    layers = getattr(getattr(runtime, "model", None), "layers", None)
    if not isinstance(layers, nn.ModuleList) or len(layers) != len(codebook):
        raise Qwen3AttentionSurgeryStateError(
            "Qwen3 decoder depth does not match the attention codebook."
        )
    for layer_index, layer in enumerate(layers):
        view = codebook[layer_index]
        attention = layer.self_attn
        if any(
            getattr(attention, role) is not getattr(view, role)
            for role in ("q_proj", "k_proj", "v_proj", "o_proj")
        ):
            raise Qwen3AttentionSurgeryStateError(
                "A partial Q/K/V/O attention-codebook installation was detected."
            )
    assignments = _global_assignments(registry)
    if set(int(item) for item in assignments.reshape(-1).tolist()) != set(
        range(_prototype_count(registry))
    ):
        raise Qwen3AttentionSurgeryStateError(
            "Every physical attention prototype must be actively assigned."
        )
    if any(parameter.grad is not None for parameter in registry.parameters()):
        raise Qwen3AttentionSurgeryStateError(
            "Attention surgery requires a zero-gradient registry."
        )
    return codebook, registry


def qwen3_attention_runtime_snapshot(
    runtime: nn.Module,
) -> Qwen3AttentionRuntimeSnapshot:
    """Capture exact physical storage and coupled assignment provenance."""

    _, registry = _validated_runtime(runtime)
    return _registry_snapshot(registry)


def qwen3_attention_codebook_storage_bytes(runtime: nn.Module) -> int:
    return qwen3_attention_runtime_snapshot(runtime).tensor_bytes


@dataclass(frozen=True, slots=True)
class _InstalledRuntime:
    codebook: GlobalAttentionCodebook
    snapshot: Qwen3AttentionRuntimeSnapshot
    aggregate_report: Qwen3CodebookParameterReport


def _wire_projection(layer: nn.Module, name: str, value: nn.Module) -> None:
    setattr(layer.self_attn, name, value)


def _force_install_attention_codebook(
    runtime: nn.Module, codebook: GlobalAttentionCodebook
) -> None:
    setattr(runtime, "attention_codebook", codebook)
    layers = runtime.model.layers
    for layer_index, layer in enumerate(layers):
        view = codebook[layer_index]
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(layer.self_attn, name, getattr(view, name))


def _install_attention_codebook(
    runtime: nn.Module, codebook: GlobalAttentionCodebook
) -> None:
    previous, _ = _validated_runtime(runtime)
    previous_report = qwen3_codebook_parameter_report(runtime)
    if len(runtime.model.layers) != len(codebook):
        raise Qwen3AttentionSurgeryStateError(
            "Replacement attention codebook depth does not match Qwen3."
        )
    try:
        setattr(runtime, "attention_codebook", codebook)
        for layer_index, layer in enumerate(runtime.model.layers):
            view = codebook[layer_index]
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                _wire_projection(layer, name, getattr(view, name))
        synchronized_report = qwen3_codebook_parameter_report(runtime)
        installed, _ = _validated_runtime(runtime)
        if installed is not codebook:
            raise Qwen3AttentionSurgeryStateError(
                "Replacement attention codebook was not installed by identity."
            )
        if synchronized_report.attention_unique_parameter_count != (
            codebook.report.unique_learned_parameters
        ):
            raise Qwen3AttentionSurgeryStateError(
                "Aggregate Qwen accounting did not synchronize after installation."
            )
    except Exception:
        _force_install_attention_codebook(runtime, previous)
        setattr(runtime, _QWEN3_REPORT_ATTRIBUTE, previous_report)
        qwen3_codebook_parameter_report(runtime)
        raise


def _capture_installed(runtime: nn.Module) -> _InstalledRuntime:
    codebook, registry = _validated_runtime(runtime)
    return _InstalledRuntime(
        codebook,
        _registry_snapshot(registry),
        qwen3_codebook_parameter_report(runtime),
    )


def _restore_installed(runtime: nn.Module, installed: _InstalledRuntime) -> bool:
    _force_install_attention_codebook(runtime, installed.codebook)
    setattr(runtime, _QWEN3_REPORT_ATTRIBUTE, installed.aggregate_report)
    synchronized = qwen3_codebook_parameter_report(runtime)
    codebook, registry = _validated_runtime(runtime)
    return (
        codebook is installed.codebook
        and _registry_snapshot(registry) == installed.snapshot
        and synchronized == installed.aggregate_report
    )


def _derived_weight_features(
    registry: AttentionCodebookRegistry, slot: int, dimensions: int
) -> tuple[float, ...]:
    vector = torch.cat(
        [
            value.detach().reshape(-1)
            for value in _prototype_bundle_tensors(registry, slot)
        ]
    ).to(device="cpu", dtype=torch.float64)
    count = min(
        _positive_int(dimensions, "derived_weight_dimensions"), int(vector.numel())
    )
    if count == int(vector.numel()):
        selected = vector
    else:
        indices = torch.div(
            torch.arange(count, dtype=torch.int64) * int(vector.numel()),
            count,
            rounding_mode="floor",
        )
        selected = vector.index_select(0, indices)
    return tuple(float(item) for item in selected.tolist())


def _features_to_nodes(
    runtime: nn.Module,
    features: Sequence[Qwen3AttentionSurgeryFeature],
    bounds: Qwen3AttentionFeatureBounds,
) -> tuple[SurgeryNode, ...]:
    _, registry = _validated_runtime(runtime)
    if len(features) > bounds.max_nodes:
        raise Qwen3AttentionSurgeryShapeError(
            f"features exceeds max_nodes={bounds.max_nodes}."
        )
    groups = int(_global_assignments(registry).shape[1])
    active = {
        qwen3_attention_group_node_id(*divmod(int(source), groups)): slot
        for slot, source in enumerate(_source_indices(registry).tolist())
    }
    expected_dimensions: dict[str, int] = {}
    total_values = 0
    seen: set[Qwen3AttentionNodeId] = set()
    nodes: list[SurgeryNode] = []
    for feature in sorted(features, key=lambda item: item.node_id):
        node = feature.node_id
        if node in seen:
            raise Qwen3AttentionSurgeryShapeError(f"Duplicate feature node {node!r}.")
        seen.add(node)
        if node not in active:
            raise Qwen3AttentionSurgeryStateError(
                f"Feature {node!r} is not an active real attention prototype."
            )
        weight = feature.weight_features or _derived_weight_features(
            registry, active[node], bounds.derived_weight_dimensions
        )
        channels: dict[str, tuple[float, ...]] = {
            "weight": weight,
            "activation": feature.activation_features,
            "jvp": feature.jvp_features,
            "output": feature.output_features,
        }
        limits = {
            "weight": bounds.max_weight_dimensions,
            "activation": bounds.max_activation_dimensions,
            "jvp": bounds.max_jvp_dimensions,
            "output": bounds.max_output_dimensions,
        }
        if feature.domain_features is not None:
            channels["domain"] = feature.domain_features
            limits["domain"] = bounds.max_domain_dimensions
        for name, values in channels.items():
            if len(values) > limits[name]:
                raise Qwen3AttentionSurgeryShapeError(
                    f"{node!r} {name} feature dimension exceeds {limits[name]}."
                )
            expected = expected_dimensions.setdefault(name, len(values))
            if expected != len(values):
                raise Qwen3AttentionSurgeryShapeError(
                    f"All {name} feature vectors must have dimension {expected}."
                )
            total_values += len(values)
        nodes.append(
            SurgeryNode(
                node_id=node,
                weight=weight,
                activation=feature.activation_features,
                rollout_jvp=feature.jvp_features,
                output_dual=feature.output_features,
                domain=feature.domain_features,
                importance=feature.importance,
                parameter_bytes=_prototype_bytes(registry),
                anchor=feature.anchor,
            )
        )
    if set(active) != seen:
        missing = sorted(set(active) - seen)
        raise Qwen3AttentionSurgeryShapeError(
            f"features must cover every active attention prototype; missing {missing[:4]!r}."
        )
    if total_values > bounds.max_total_values:
        raise Qwen3AttentionSurgeryShapeError(
            f"Feature payload has {total_values} values, exceeding {bounds.max_total_values}."
        )
    return tuple(nodes)


def _least_squares_scale(source: Tensor, target: Tensor) -> float:
    source64 = source.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    target64 = target.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    denominator = float(torch.dot(target64, target64))
    if denominator <= torch.finfo(torch.float64).eps:
        return 1.0
    result = float(torch.dot(source64, target64)) / denominator
    return result if math.isfinite(result) else 1.0


def _prototype_affine_block(
    registry: AttentionCodebookRegistry,
    role: str,
    slot: int,
) -> tuple[Tensor, bool]:
    weight = _flat_prototype_tensor(registry, f"prototype_{role}_weight")
    if weight is None:  # pragma: no cover - every role weight is mandatory
        raise Qwen3AttentionSurgeryStateError(f"Missing {role} prototype weight.")
    block = weight[slot].detach()
    bias = _flat_prototype_tensor(registry, f"prototype_{role}_bias")
    if bias is None:
        return block, False
    return torch.cat((block, bias[slot].detach().unsqueeze(-1)), dim=-1), True


def _post_norm_affine_direction(
    block: Tensor,
    norm_weight: Tensor,
    head_dim: int,
    epsilon: float,
) -> Tensor:
    heads = int(block.shape[0]) // head_dim
    reshaped = block.detach().to(device="cpu", dtype=torch.float64).reshape(
        heads, head_dim, -1
    )
    rms = torch.sqrt(torch.mean(reshaped.square(), dim=1, keepdim=True) + epsilon)
    normalized = reshaped / rms
    normalized *= norm_weight.detach().to(device="cpu", dtype=torch.float64).reshape(
        1, head_dim, 1
    )
    return normalized.reshape(-1)


def _normalized_tensor_residual(source: Tensor, target: Tensor) -> float:
    source64 = source.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    target64 = target.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    residual = source64 - target64
    denominator = float(torch.mean(source64.square())) + 1e-12
    return float(torch.mean(residual.square())) / denominator


def _automatic_projection(
    runtime: nn.Module,
    registry: AttentionCodebookRegistry,
    source_slot: int,
    target_slot: int,
    source_id: Qwen3AttentionNodeId,
    target_id: Qwen3AttentionNodeId,
) -> Qwen3AttentionProjection:
    source_layer = source_id[2]
    target_layer = target_id[2]
    source_attention = runtime.model.layers[source_layer].self_attn
    target_attention = runtime.model.layers[target_layer].self_attn
    epsilon = float(getattr(runtime.config, "rms_norm_eps", 1e-6))
    q_source, q_bias = _prototype_affine_block(registry, "q", source_slot)
    q_target, q_target_bias = _prototype_affine_block(registry, "q", target_slot)
    k_source, k_bias = _prototype_affine_block(registry, "k", source_slot)
    k_target, k_target_bias = _prototype_affine_block(registry, "k", target_slot)
    q_source_direction = _post_norm_affine_direction(
        q_source, source_attention.q_norm.weight, registry.head_dim, epsilon
    )
    q_target_direction = _post_norm_affine_direction(
        q_target, target_attention.q_norm.weight, registry.head_dim, epsilon
    )
    k_source_direction = _post_norm_affine_direction(
        k_source, source_attention.k_norm.weight, registry.head_dim, epsilon
    )
    k_target_direction = _post_norm_affine_direction(
        k_target, target_attention.k_norm.weight, registry.head_dim, epsilon
    )
    qk_residual = 0.5 * (
        _normalized_tensor_residual(q_source_direction, q_target_direction)
        + _normalized_tensor_residual(k_source_direction, k_target_direction)
    )
    v_source, v_bias = _prototype_affine_block(registry, "v", source_slot)
    v_target, v_target_bias = _prototype_affine_block(registry, "v", target_slot)
    o_source, _ = _prototype_affine_block(registry, "o", source_slot)
    o_target, _ = _prototype_affine_block(registry, "o", target_slot)
    return Qwen3AttentionProjection(
        source_id,
        target_id,
        1.0,
        1.0,
        _least_squares_scale(v_source, v_target),
        _least_squares_scale(o_source, o_target),
        qk_normalized_residual=qk_residual,
        bias_included=any(
            (q_bias, q_target_bias, k_bias, k_target_bias, v_bias, v_target_bias)
        ),
        fit_space="post-qk-rmsnorm-affine-direction-plus-v/o-affine",
    )


def _normalized_projection_residual(
    source: tuple[float, ...], target: tuple[float, ...], coefficient: float
) -> float:
    source_array = np.asarray(source, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    residual = source_array - coefficient * target_array
    denominator = float(np.mean(np.square(source_array), dtype=np.float64)) + 1e-12
    return float(np.mean(np.square(residual), dtype=np.float64)) / denominator


@dataclass(frozen=True, slots=True)
class Qwen3AttentionCandidateScore:
    candidate: MergeCandidate
    projection: Qwen3AttentionProjection
    projection_damage: float
    prototype_bytes_saved: int
    scale_allocation_bytes: int
    physical_bytes_saved: int
    savings_reward: float
    total_score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_sha256": self.candidate.candidate_sha256,
            "projection": self.projection.as_dict(),
            "projection_damage": self.projection_damage,
            "prototype_bytes_saved": self.prototype_bytes_saved,
            "scale_allocation_bytes": self.scale_allocation_bytes,
            "physical_bytes_saved": self.physical_bytes_saved,
            "savings_reward": self.savings_reward,
            "total_score": self.total_score,
        }


def _requires_scales(projection: Qwen3AttentionProjection) -> bool:
    return any(value != 1.0 for value in projection.values)


def _copy_requires_grad(
    source: AttentionCodebookRegistry,
    target: RaggedAttentionCodebookRegistry,
    *,
    scales_created: bool,
) -> None:
    source_parameters = dict(source.named_parameters())
    for name, parameter in target.named_parameters():
        old = source_parameters.get(name)
        parameter.requires_grad_(
            False
            if name == "role_scales" and scales_created
            else True
            if old is None
            else old.requires_grad
        )


@dataclass(frozen=True, slots=True)
class Qwen3AttentionCompactionWork:
    """Conservative live-storage and copy-work bound for one compaction."""

    source_tensor_bytes: int
    result_tensor_bytes: int
    copied_tensor_bytes: int
    maximum_staging_bytes: int
    peak_live_tensor_bytes_bound: int
    full_registry_clone_count: int = 0
    prototype_copy_passes: int = 1

    def as_dict(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name)) for name in self.__dataclass_fields__
        }


def _stream_select_slots(value: Tensor | None, slots: Sequence[int], dim: int) -> Tensor | None:
    if value is None:
        return None
    shape = list(value.shape)
    shape[dim] = len(slots)
    result = torch.empty(shape, dtype=value.dtype, device=value.device)
    with torch.inference_mode():
        for destination, source in enumerate(slots):
            result.select(dim, destination).copy_(value.detach().select(dim, source))
    return result


def _compaction_work(
    source: AttentionCodebookRegistry,
    result: AttentionCodebookRegistry,
) -> Qwen3AttentionCompactionWork:
    source_bytes = sum(_tensor_bytes(value) for value in source.state_dict().values())
    result_bytes = sum(_tensor_bytes(value) for value in result.state_dict().values())
    copied_names = (
        "prototype_q_weight",
        "prototype_k_weight",
        "prototype_v_weight",
        "prototype_o_weight",
        "prototype_q_bias",
        "prototype_k_bias",
        "prototype_v_bias",
    )
    copied = sum(
        _tensor_bytes(getattr(result, name, None)) for name in copied_names
    )
    maximum_staging = max(
        (
            _tensor_bytes(value[0])
            for name in copied_names
            if (value := getattr(result, name, None)) is not None
        ),
        default=0,
    )
    return Qwen3AttentionCompactionWork(
        source_tensor_bytes=source_bytes,
        result_tensor_bytes=result_bytes,
        copied_tensor_bytes=copied,
        maximum_staging_bytes=maximum_staging,
        peak_live_tensor_bytes_bound=source_bytes + result_bytes + maximum_staging,
    )


def _compacted_attention_codebook(
    runtime: nn.Module,
    graph_snapshot: SurgerySnapshot,
    projection_for_pair: Callable[
        [Qwen3AttentionNodeId, Qwen3AttentionNodeId], Qwen3AttentionProjection
    ],
) -> tuple[GlobalAttentionCodebook, tuple[Qwen3AttentionProjection, ...]]:
    current, registry = _validated_runtime(runtime)
    groups = int(_global_assignments(registry).shape[1])
    sources = tuple(int(item) for item in _source_indices(registry).tolist())
    node_by_slot = tuple(
        qwen3_attention_group_node_id(*divmod(source, groups)) for source in sources
    )
    slot_by_node = {node: slot for slot, node in enumerate(node_by_slot)}
    redirects = {slot: slot for slot in range(len(sources))}
    for source_node, target_node in graph_snapshot.assignment_pairs:
        source = _node_ref(source_node, "graph assignment source")
        target = _node_ref(target_node, "graph assignment target")
        if source not in slot_by_node or target not in slot_by_node:
            raise Qwen3AttentionSurgeryStateError(
                "Graph assignment references a non-active attention prototype."
            )
        redirects[slot_by_node[source]] = slot_by_node[target]
    for slot, target in redirects.items():
        if redirects[target] != target:
            raise Qwen3AttentionSurgeryStateError(
                f"Attention redirect {slot}->{target} does not end at a real medoid."
            )
    survivors = tuple(sorted(set(redirects.values())))
    if len(survivors) >= len(sources):
        raise Qwen3AttentionSurgeryStateError(
            "Attention proposal did not remove physical coupled-prototype storage."
        )
    new_slot = {old: index for index, old in enumerate(survivors)}
    old_assignments = _global_assignments(registry)
    flattened = [
        new_slot[redirects[int(slot)]] for slot in old_assignments.reshape(-1).tolist()
    ]
    assignments = torch.tensor(flattened, dtype=torch.int64).reshape_as(old_assignments)
    projections: dict[
        tuple[Qwen3AttentionNodeId, Qwen3AttentionNodeId], Qwen3AttentionProjection
    ] = {}
    for source_slot, target_slot in redirects.items():
        if source_slot == target_slot:
            continue
        source_id = node_by_slot[source_slot]
        target_id = node_by_slot[target_slot]
        projections[(source_id, target_id)] = projection_for_pair(source_id, target_id)
    old_scales = registry.role_scales
    needs_scales = old_scales is not None or any(
        _requires_scales(item) for item in projections.values()
    )
    scales = None
    if needs_scales:
        if old_scales is None:
            scales = torch.ones(
                (*old_assignments.shape, 4),
                device=registry.prototype_q_weight.device,
                dtype=registry.prototype_q_weight.dtype,
            )
        else:
            scales = old_scales.detach().clone()
        for layer_index, row in enumerate(old_assignments.tolist()):
            for group_index, source_slot in enumerate(row):
                target_slot = redirects[int(source_slot)]
                if target_slot == int(source_slot):
                    continue
                projection = projections[
                    (node_by_slot[int(source_slot)], node_by_slot[target_slot])
                ]
                coefficient = torch.tensor(
                    projection.values, device=scales.device, dtype=scales.dtype
                )
                scales[layer_index, group_index] *= coefficient
    def selected_tensor(name: str) -> Tensor | None:
        value = _flat_prototype_tensor(registry, name)
        return _stream_select_slots(value, survivors, 0)

    q_weight = selected_tensor("prototype_q_weight")
    k_weight = selected_tensor("prototype_k_weight")
    v_weight = selected_tensor("prototype_v_weight")
    o_weight = selected_tensor("prototype_o_weight")
    if any(value is None for value in (q_weight, k_weight, v_weight, o_weight)):
        raise Qwen3AttentionSurgeryStateError(
            "A coupled attention proposal lost one of Q/K/V/O."
        )
    new_registry = RaggedAttentionCodebookRegistry(
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        assignments,
        _stream_select_slots(
            _source_indices(registry).to(device=registry.prototype_q_weight.device),
            survivors,
            0,
        ),
        registry.group_permutations,
        prototype_q_bias=selected_tensor("prototype_q_bias"),
        prototype_k_bias=selected_tensor("prototype_k_bias"),
        prototype_v_bias=selected_tensor("prototype_v_bias"),
        layer_o_bias=registry.layer_o_bias,
        role_scales=scales,
        copy_tensors=False,
    )
    _copy_requires_grad(
        registry, new_registry, scales_created=old_scales is None and scales is not None
    )
    new_registry.compaction_work = _compaction_work(registry, new_registry)
    new_registry.train(registry.training)
    result = GlobalAttentionCodebook(new_registry)
    result.train(current.training)
    return result, tuple(projections[pair] for pair in sorted(projections))


@dataclass(frozen=True, slots=True)
class Qwen3AttentionRuntimeAudit:
    """Constant-size physical runtime evidence retained in bounded history."""

    prototype_count: int
    tensor_bytes: int
    prototype_pool_bytes: int
    scale_bytes: int
    registry_sha256: str

    @classmethod
    def from_snapshot(
        cls, snapshot: Qwen3AttentionRuntimeSnapshot
    ) -> Qwen3AttentionRuntimeAudit:
        return cls(
            snapshot.prototype_count,
            snapshot.tensor_bytes,
            snapshot.prototype_pool_bytes,
            snapshot.scale_bytes,
            snapshot.registry_sha256,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "prototype_count": self.prototype_count,
            "tensor_bytes": self.tensor_bytes,
            "prototype_pool_bytes": self.prototype_pool_bytes,
            "scale_bytes": self.scale_bytes,
            "registry_sha256": self.registry_sha256,
        }


@dataclass(frozen=True, slots=True)
class Qwen3AttentionSurgeryEvaluationContext:
    runtime: nn.Module
    round_index: int
    evidence_epoch: int
    target_bytes: int
    parent_graph_snapshot: SurgerySnapshot
    proposed_graph_snapshot: SurgerySnapshot
    graph_transaction: SurgeryTransaction
    parent_runtime_snapshot: Qwen3AttentionRuntimeSnapshot
    proposed_runtime_snapshot: Qwen3AttentionRuntimeSnapshot
    projections: tuple[Qwen3AttentionProjection, ...]


@dataclass(frozen=True, slots=True)
class Qwen3AttentionSurgeryRefreshContext:
    runtime: nn.Module
    round_index: int
    accepted_transaction_index: int
    evidence_epoch: int
    target_bytes: int
    current_bytes: int
    active_node_ids: tuple[Qwen3AttentionNodeId, ...]
    graph_transaction: SurgeryTransaction


@dataclass(frozen=True, slots=True)
class Qwen3AttentionCandidateSelectionContext:
    graph: AdaptiveSurgeryGraph
    discovered_candidates: tuple[MergeCandidate, ...]
    scored_candidates: tuple[Qwen3AttentionCandidateScore, ...]
    target_bytes: int
    current_bytes: int
    max_nonconflicting_batch_size: int
    evidence_epoch: int


@dataclass(frozen=True, slots=True)
class Qwen3AttentionSurgeryDiagnostics:
    attempted_transactions: int
    accepted_transactions: int
    rejected_transactions: int
    candidate_batches: int
    bisections: int
    feature_refreshes: int
    evidence_epoch: int
    graph_rounds: int
    retained_history: int
    history_limit: int
    optimizer_steps: int = 0
    backward_calls: int = 0

    @property
    def zero_gradient(self) -> bool:
        return self.optimizer_steps == 0 and self.backward_calls == 0

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__} | {
            "zero_gradient": self.zero_gradient
        }


@dataclass(frozen=True, slots=True)
class Qwen3AttentionSurgeryTransaction:
    transaction_index: int
    round_index: int
    evidence_epoch_before: int
    evidence_epoch_after: int
    accepted: bool
    decision: EvaluationDecision
    graph_transaction_sha256: str
    action_sha256s: tuple[str, ...]
    candidate_sha256s: tuple[str, ...]
    parent_runtime: Qwen3AttentionRuntimeAudit
    proposed_runtime: Qwen3AttentionRuntimeAudit
    committed_runtime: Qwen3AttentionRuntimeAudit
    assignment_change_count: int
    assignment_changes_sha256: str
    projections: tuple[Qwen3AttentionProjection, ...]
    physical_bytes_saved: int
    rollback_exact: bool | None
    measured_improvement_claim: bool

    @property
    def reverted(self) -> bool:
        return not self.accepted

    @property
    def rollback_scope(self) -> str:
        return "owned-attention-codebook-installation"

    def as_dict(self) -> dict[str, Any]:
        return {
            "transaction_index": self.transaction_index,
            "round_index": self.round_index,
            "evidence_epoch_before": self.evidence_epoch_before,
            "evidence_epoch_after": self.evidence_epoch_after,
            "accepted": self.accepted,
            "reverted": self.reverted,
            "decision": self.decision.as_dict(),
            "graph_transaction_sha256": self.graph_transaction_sha256,
            "action_sha256s": list(self.action_sha256s),
            "candidate_sha256s": list(self.candidate_sha256s),
            "parent_runtime": self.parent_runtime.as_dict(),
            "proposed_runtime": self.proposed_runtime.as_dict(),
            "committed_runtime": self.committed_runtime.as_dict(),
            "assignment_change_count": self.assignment_change_count,
            "assignment_changes_sha256": self.assignment_changes_sha256,
            "projections": [item.as_dict() for item in self.projections],
            "physical_bytes_saved": self.physical_bytes_saved,
            "rollback_exact": self.rollback_exact,
            "measured_improvement_claim": self.measured_improvement_claim,
            "rollback_scope": self.rollback_scope,
        }


@dataclass(frozen=True, slots=True)
class Qwen3AttentionSurgeryResult:
    runtime: nn.Module
    target_bytes: int
    initial_runtime_snapshot: Qwen3AttentionRuntimeSnapshot
    final_runtime_snapshot: Qwen3AttentionRuntimeSnapshot
    transactions: tuple[Qwen3AttentionSurgeryTransaction, ...]
    diagnostics: Qwen3AttentionSurgeryDiagnostics
    target_reached: bool

    @property
    def bytes_saved(self) -> int:
        return (
            self.initial_runtime_snapshot.tensor_bytes
            - self.final_runtime_snapshot.tensor_bytes
        )

    @property
    def zero_gradient(self) -> bool:
        return self.diagnostics.zero_gradient

    @property
    def immutable_rollback_verified(self) -> bool:
        return all(
            transaction.accepted or transaction.rollback_exact is True
            for transaction in self.transactions
        )

    @property
    def owned_registry_rollback_verified(self) -> bool:
        return self.immutable_rollback_verified

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "qwen3-adaptive-coupled-gqa-surgery",
            "target_bytes": self.target_bytes,
            "target_reached": self.target_reached,
            "initial_runtime": self.initial_runtime_snapshot.as_dict(),
            "final_runtime": self.final_runtime_snapshot.as_dict(),
            "bytes_saved": self.bytes_saved,
            "transactions": [item.as_dict() for item in self.transactions],
            "diagnostics": self.diagnostics.as_dict(),
            "zero_gradient": self.zero_gradient,
            "immutable_rollback_verified": self.immutable_rollback_verified,
            "owned_registry_rollback_verified": self.owned_registry_rollback_verified,
            "rollback_scope": "owned-attention-codebook-installation",
            "quality_improvement_claimed_by_bridge": False,
        }

    @property
    def report(self) -> dict[str, Any]:
        return self.as_dict()


def _decision(
    value: Any,
    context: Qwen3AttentionSurgeryEvaluationContext,
) -> tuple[EvaluationDecision, bool]:
    if isinstance(value, EvaluationDecision):
        decision = value
        raw_metrics = dict(value.metrics)
    elif isinstance(value, bool):
        return EvaluationDecision(value), False
    elif isinstance(value, Mapping):
        accepted = value.get("accepted", value.get("accept"))
        if not isinstance(accepted, bool):
            raise Qwen3AttentionSurgeryShapeError(
                "gate mappings require a boolean accepted or accept field."
            )
        reason_value = value.get("reason", "")
        reason = "" if reason_value is None else str(reason_value)
        raw_metrics = {
            str(key): item
            for key, item in value.items()
            if key not in {"accepted", "accept", "reason"}
            and isinstance(item, (str, int, float, bool, type(None)))
            and not (isinstance(item, float) and not math.isfinite(item))
        }
        decision = EvaluationDecision(accepted, reason=reason, metrics=raw_metrics)
    else:
        raise Qwen3AttentionSurgeryShapeError(
            "gate must return bool, EvaluationDecision, or a mapping."
        )

    normalized_reason = decision.reason.strip().lower()
    claim = any(
        raw_metrics.get(name) is True
        for name in (
            "improved",
            "improvement",
            "quality_improved",
            "model_improved",
            "better_than_parent",
            "outperformed_parent",
        )
    ) or any(
        phrase in normalized_reason
        for phrase in ("quality improved", "better than parent", "outperformed parent")
    )
    if not claim:
        return decision, False
    required = {
        "parent_score",
        "candidate_score",
        "higher_is_better",
        "evaluation_contract_sha256",
        "parent_runtime_sha256",
    }
    if not required.issubset(raw_metrics):
        raise Qwen3AttentionSurgeryStateError(
            "An improvement claim requires measured scores, direction, an evaluation "
            "contract hash, and the exact parent runtime hash."
        )
    contract_hash = raw_metrics["evaluation_contract_sha256"]
    parent_hash = raw_metrics["parent_runtime_sha256"]
    if (
        not isinstance(contract_hash, str)
        or len(contract_hash) != 64
        or any(character not in "0123456789abcdef" for character in contract_hash.lower())
    ):
        raise Qwen3AttentionSurgeryShapeError(
            "evaluation_contract_sha256 must be a 64-character SHA-256 hex digest."
        )
    if parent_hash != context.parent_runtime_snapshot.registry_sha256:
        raise Qwen3AttentionSurgeryStateError(
            "Improvement claim parent_runtime_sha256 does not match the gated parent."
        )
    parent = _finite_float(raw_metrics["parent_score"], "parent_score")
    candidate = _finite_float(raw_metrics["candidate_score"], "candidate_score")
    higher = raw_metrics["higher_is_better"]
    if not isinstance(higher, bool):
        raise Qwen3AttentionSurgeryShapeError("higher_is_better must be boolean.")
    measured = candidate >= parent if higher else candidate <= parent
    if not measured:
        raise Qwen3AttentionSurgeryStateError(
            "The supplied parent-relative measurements contradict the improvement claim."
        )
    return decision, True


def _invoke_gate(
    gate: Gate | None, context: Qwen3AttentionSurgeryEvaluationContext
) -> tuple[EvaluationDecision, bool]:
    if gate is None:
        return EvaluationDecision.accept(reason="no external quality claim"), False
    callable_value = getattr(gate, "evaluate", gate)
    if not callable(callable_value):
        raise Qwen3AttentionSurgeryShapeError(
            "gate must be callable or expose evaluate()."
        )
    try:
        signature = inspect.signature(callable_value)
    except (TypeError, ValueError):
        return _decision(callable_value(context), context)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    ]
    if not positional:
        return _decision(callable_value(), context)
    if len(positional) == 1:
        return _decision(callable_value(context), context)
    return _decision(callable_value(context.runtime, context), context)


def _assignment_changes(
    before: Qwen3AttentionRuntimeSnapshot,
    after: Qwen3AttentionRuntimeSnapshot,
) -> tuple[tuple[int, int, int, int], ...]:
    if (
        before.logical_layers != after.logical_layers
        or before.key_value_heads != after.key_value_heads
    ):
        raise Qwen3AttentionSurgeryStateError(
            "Attention surgery changed the logical layer/group topology."
        )
    changes = []
    for layer_index in range(before.logical_layers):
        for group_index in range(before.key_value_heads):
            old = before.assignment_source_indices[layer_index][group_index]
            new = after.assignment_source_indices[layer_index][group_index]
            if old != new:
                changes.append((layer_index, group_index, old, new))
    return tuple(changes)


@dataclass(frozen=True, slots=True)
class _BridgeCheckpoint:
    installed: _InstalledRuntime
    graph: AdaptiveSurgeryGraph
    graph_checkpoint: Any
    features: tuple[Qwen3AttentionSurgeryFeature, ...]
    projections: tuple[Qwen3AttentionProjection, ...]
    transactions: tuple[Qwen3AttentionSurgeryTransaction, ...]
    counters: tuple[int, ...]


class Qwen3AttentionSurgeryBridge:
    """Bounded, atomic surgery over complete Qwen3 GQA group paths."""

    def __init__(
        self,
        runtime: nn.Module,
        features: Sequence[Qwen3AttentionSurgeryFeature],
        *,
        projections: (Sequence[Qwen3AttentionProjection] | Mapping[Any, Any]) = (),
        projection_coefficients: (
            Sequence[Qwen3AttentionProjection] | Mapping[Any, Any] | None
        ) = None,
        feature_bounds: Qwen3AttentionFeatureBounds | None = None,
        score_weights: ScoreWeights | None = None,
        projection_count: int = 8,
        neighbor_window: int = 4,
        max_candidates: int = 64,
        medoid_candidate_limit: int = 32,
        max_nonconflicting_batch_size: int = 1,
        transaction_history_limit: int = 32,
        candidate_selector: CandidateSelector | None = None,
    ) -> None:
        _validated_runtime(runtime)
        if projection_coefficients is not None:
            if projections:
                raise Qwen3AttentionSurgeryShapeError(
                    "Provide projections or projection_coefficients, not both."
                )
            projections = projection_coefficients
        self.runtime = runtime
        self.feature_bounds = feature_bounds or Qwen3AttentionFeatureBounds()
        if not isinstance(self.feature_bounds, Qwen3AttentionFeatureBounds):
            raise Qwen3AttentionSurgeryShapeError(
                "feature_bounds must be Qwen3AttentionFeatureBounds."
            )
        self.score_weights = score_weights or ScoreWeights()
        self.projection_count = _positive_int(projection_count, "projection_count")
        self.neighbor_window = _positive_int(neighbor_window, "neighbor_window")
        self.max_candidates = _positive_int(max_candidates, "max_candidates")
        self.medoid_candidate_limit = _positive_int(
            medoid_candidate_limit, "medoid_candidate_limit"
        )
        self.max_nonconflicting_batch_size = _positive_int(
            max_nonconflicting_batch_size, "max_nonconflicting_batch_size"
        )
        self.transaction_history_limit = _positive_int(
            transaction_history_limit, "transaction_history_limit"
        )
        if candidate_selector is not None and not callable(
            getattr(candidate_selector, "select", candidate_selector)
        ):
            raise Qwen3AttentionSurgeryShapeError(
                "candidate_selector must be callable or expose select()."
            )
        self._candidate_selector = candidate_selector
        self._features = _feature_sequence(features)
        self._projections = _projection_sequence(projections)
        self._validate_projection_nodes(self._features, self._projections)
        self._graph = self._build_graph(self._features, self._projections)
        self._transactions: list[Qwen3AttentionSurgeryTransaction] = []
        self._attempt_count = 0
        self._accepted_count = 0
        self._rejected_count = 0
        self._candidate_batches = 0
        self._bisections = 0
        self._feature_refreshes = 0
        self._evidence_epoch = 0
        self._graph_rounds = 1
        self._has_run = False

    @property
    def graph(self) -> AdaptiveSurgeryGraph:
        return self._graph

    @property
    def features(self) -> tuple[Qwen3AttentionSurgeryFeature, ...]:
        return self._features

    @property
    def projections(self) -> tuple[Qwen3AttentionProjection, ...]:
        return self._projections

    @property
    def transactions(self) -> tuple[Qwen3AttentionSurgeryTransaction, ...]:
        return tuple(self._transactions)

    @property
    def evidence_epoch(self) -> int:
        return self._evidence_epoch

    @property
    def runtime_snapshot(self) -> Qwen3AttentionRuntimeSnapshot:
        return qwen3_attention_runtime_snapshot(self.runtime)

    @property
    def diagnostics(self) -> Qwen3AttentionSurgeryDiagnostics:
        return Qwen3AttentionSurgeryDiagnostics(
            attempted_transactions=self._attempt_count,
            accepted_transactions=self._accepted_count,
            rejected_transactions=self._rejected_count,
            candidate_batches=self._candidate_batches,
            bisections=self._bisections,
            feature_refreshes=self._feature_refreshes,
            evidence_epoch=self._evidence_epoch,
            graph_rounds=self._graph_rounds,
            retained_history=len(self._transactions),
            history_limit=self.transaction_history_limit,
        )

    @staticmethod
    def _validate_projection_nodes(
        features: Sequence[Qwen3AttentionSurgeryFeature],
        projections: Sequence[Qwen3AttentionProjection],
    ) -> None:
        nodes = {feature.node_id for feature in features}
        for projection in projections:
            if projection.source not in nodes or projection.target not in nodes:
                raise Qwen3AttentionSurgeryShapeError(
                    "Projection endpoints must both be active feature nodes."
                )

    @staticmethod
    def _feature_map(
        features: Sequence[Qwen3AttentionSurgeryFeature],
    ) -> dict[Qwen3AttentionNodeId, Qwen3AttentionSurgeryFeature]:
        return {feature.node_id: feature for feature in features}

    @staticmethod
    def _projection_map(
        projections: Sequence[Qwen3AttentionProjection],
    ) -> dict[
        tuple[Qwen3AttentionNodeId, Qwen3AttentionNodeId],
        Qwen3AttentionProjection,
    ]:
        return {(item.source, item.target): item for item in projections}

    def _projection_for_pair_from(
        self,
        source: Qwen3AttentionNodeId,
        target: Qwen3AttentionNodeId,
        projections: Sequence[Qwen3AttentionProjection],
        *,
        registry: AttentionCodebookRegistry | None = None,
    ) -> Qwen3AttentionProjection:
        explicit = self._projection_map(projections).get((source, target))
        if explicit is not None:
            return explicit
        if registry is None:
            _, registry = _validated_runtime(self.runtime)
        groups = int(_global_assignments(registry).shape[1])
        slot_by_node = {
            qwen3_attention_group_node_id(*divmod(int(value), groups)): slot
            for slot, value in enumerate(_source_indices(registry).tolist())
        }
        try:
            source_slot = slot_by_node[source]
            target_slot = slot_by_node[target]
        except KeyError as error:
            raise Qwen3AttentionSurgeryStateError(
                "Projection pair references a non-active prototype."
            ) from error
        return _automatic_projection(
            self.runtime,
            registry,
            source_slot,
            target_slot,
            source,
            target,
        )

    def _projection_damage(
        self,
        projection: Qwen3AttentionProjection,
        features: Sequence[Qwen3AttentionSurgeryFeature],
    ) -> float:
        by_node = self._feature_map(features)
        source = by_node[projection.source]
        target = by_node[projection.target]
        activation_coefficient = math.fsum(projection.values[:3]) / 3.0
        jvp_coefficient = projection.o
        activation = (
            projection.activation_residual
            if projection.activation_residual is not None
            else _normalized_projection_residual(
                source.activation_features,
                target.activation_features,
                activation_coefficient,
            )
        )
        jvp = (
            projection.jvp_residual
            if projection.jvp_residual is not None
            else _normalized_projection_residual(
                source.jvp_features, target.jvp_features, jvp_coefficient
            )
        )
        scale_penalty = (
            math.fsum(math.log1p(abs(value - 1.0)) ** 2 for value in projection.values)
            / 4.0
        )
        qk_geometry = (
            projection.qk_normalized_residual
            if projection.qk_normalized_residual is not None
            else math.log1p(abs(projection.q * projection.k - 1.0)) ** 2 + 1.0
        )
        return float(activation + jvp + 0.05 * scale_penalty + 0.05 * qk_geometry)

    def _build_graph(
        self,
        features: Sequence[Qwen3AttentionSurgeryFeature],
        projections: Sequence[Qwen3AttentionProjection],
    ) -> AdaptiveSurgeryGraph:
        nodes = _features_to_nodes(self.runtime, features, self.feature_bounds)
        _, evidence_registry = _validated_runtime(self.runtime)

        def projection_damage(binding, _source, _target, _distances):
            source = _node_ref(binding.source)
            target = _node_ref(binding.target)
            projection = self._projection_for_pair_from(
                source,
                target,
                projections,
                registry=evidence_registry,
            )
            return self._projection_damage(projection, features)

        return AdaptiveSurgeryGraph(
            nodes,
            score_weights=self.score_weights,
            projection_count=self.projection_count,
            neighbor_window=self.neighbor_window,
            max_candidates=self.max_candidates,
            medoid_candidate_limit=self.medoid_candidate_limit,
            history_limit=self.transaction_history_limit,
            projection_damage_callback=projection_damage,
        )

    @staticmethod
    def _removed_node(candidate: MergeCandidate) -> Qwen3AttentionNodeId:
        removed = [
            _node_ref(node)
            for node in candidate.member_ids
            if _node_ref(node) != _node_ref(candidate.resulting_medoid)
        ]
        if len(removed) != 1:
            raise Qwen3AttentionSurgeryStateError(
                "Each fresh attention candidate must remove exactly one coupled prototype."
            )
        return removed[0]

    def candidate_score(
        self, candidate: MergeCandidate
    ) -> Qwen3AttentionCandidateScore:
        if not isinstance(candidate, MergeCandidate):
            raise Qwen3AttentionSurgeryShapeError("candidate must be a MergeCandidate.")
        if candidate.parent_snapshot_sha256 != self._graph.snapshot.snapshot_sha256:
            raise Qwen3AttentionSurgeryStateError("Cannot score a stale candidate.")
        source = self._removed_node(candidate)
        target = _node_ref(candidate.resulting_medoid)
        projection = self._projection_for_pair_from(source, target, self._projections)
        damage = self._projection_damage(projection, self._features)
        _, registry = _validated_runtime(self.runtime)
        prototype_bytes = _prototype_bytes(registry)
        scale_allocation = (
            int(registry.logical_layers)
            * int(registry.key_value_heads)
            * 4
            * int(registry.prototype_q_weight.element_size())
            if registry.role_scales is None and _requires_scales(projection)
            else 0
        )
        physical = prototype_bytes - scale_allocation
        reward = physical / max(float(prototype_bytes), 1.0)
        total = (
            candidate.score.total_score
            - candidate.score.savings_reward
            + reward
            - damage
        )
        return Qwen3AttentionCandidateScore(
            candidate=candidate,
            projection=projection,
            projection_damage=damage,
            prototype_bytes_saved=prototype_bytes,
            scale_allocation_bytes=scale_allocation,
            physical_bytes_saved=physical,
            savings_reward=reward,
            total_score=total,
        )

    def propose_pair(self, left: Any, right: Any) -> Qwen3AttentionCandidateScore:
        candidate = self._graph.propose_candidate_for_pair(
            _node_ref(left, "left group"), _node_ref(right, "right group")
        )
        return self.candidate_score(candidate)

    score_pair = propose_pair

    def _graph_checkpoint(self) -> Any:
        checkpoint = getattr(self._graph, "_checkpoint", None)
        if not callable(checkpoint):
            raise Qwen3AttentionSurgeryStateError(
                "Adaptive surgery graph lacks transactional checkpointing."
            )
        return checkpoint()

    @staticmethod
    def _restore_graph(graph: AdaptiveSurgeryGraph, checkpoint: Any) -> None:
        restore = getattr(graph, "_restore_checkpoint", None)
        if not callable(restore):
            raise Qwen3AttentionSurgeryStateError(
                "Adaptive surgery graph lacks checkpoint restore."
            )
        restore(checkpoint)

    def _checkpoint(self) -> _BridgeCheckpoint:
        return _BridgeCheckpoint(
            installed=_capture_installed(self.runtime),
            graph=self._graph,
            graph_checkpoint=self._graph_checkpoint(),
            features=self._features,
            projections=self._projections,
            transactions=tuple(self._transactions),
            counters=(
                self._attempt_count,
                self._accepted_count,
                self._rejected_count,
                self._candidate_batches,
                self._bisections,
                self._feature_refreshes,
                self._evidence_epoch,
                self._graph_rounds,
            ),
        )

    def _restore_checkpoint(self, checkpoint: _BridgeCheckpoint) -> None:
        if not _restore_installed(self.runtime, checkpoint.installed):
            raise Qwen3AttentionSurgeryStateError(
                "Attention transaction rollback did not restore exact tensors."
            )
        self._graph = checkpoint.graph
        self._restore_graph(self._graph, checkpoint.graph_checkpoint)
        self._features = checkpoint.features
        self._projections = checkpoint.projections
        self._transactions = list(checkpoint.transactions)
        (
            self._attempt_count,
            self._accepted_count,
            self._rejected_count,
            self._candidate_batches,
            self._bisections,
            self._feature_refreshes,
            self._evidence_epoch,
            self._graph_rounds,
        ) = checkpoint.counters

    def _append_transaction(
        self, transaction: Qwen3AttentionSurgeryTransaction
    ) -> None:
        self._transactions.append(transaction)
        overflow = len(self._transactions) - self.transaction_history_limit
        if overflow > 0:
            del self._transactions[:overflow]

    @staticmethod
    def _is_explicit_pair(value: Any) -> bool:
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != 2
        ):
            return False
        try:
            _node_ref(value[0])
            _node_ref(value[1])
        except Qwen3AttentionSurgeryShapeError:
            return False
        return True

    @staticmethod
    def _invoke_selector(
        selector: CandidateSelector,
        context: Qwen3AttentionCandidateSelectionContext,
    ) -> Any:
        callable_value = getattr(selector, "select", selector)
        if not callable(callable_value):
            raise Qwen3AttentionSurgeryShapeError(
                "candidate_selector must be callable or expose select()."
            )
        try:
            signature = inspect.signature(callable_value)
        except (TypeError, ValueError):
            return callable_value(context)
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        ]
        if not positional:
            return callable_value()
        if len(positional) == 1:
            name = positional[0].name.lower()
            if "candidate" in name or "score" in name:
                return callable_value(context.scored_candidates)
            if "graph" in name:
                return callable_value(context.graph)
            return callable_value(context)
        return callable_value(context.discovered_candidates, context)

    def _batch_physical_bytes_saved(self, candidates: Sequence[MergeCandidate]) -> int:
        if not candidates:
            return 0
        _, registry = _validated_runtime(self.runtime)
        prototype_savings = len(candidates) * _prototype_bytes(registry)
        allocate_scales = registry.role_scales is None and any(
            _requires_scales(self.candidate_score(candidate).projection)
            for candidate in candidates
        )
        scale_bytes = (
            int(registry.logical_layers)
            * int(registry.key_value_heads)
            * 4
            * int(registry.prototype_q_weight.element_size())
            if allocate_scales
            else 0
        )
        return prototype_savings - scale_bytes

    def _selected_candidates(
        self,
        *,
        target_bytes: int,
        current_bytes: int,
        selector: CandidateSelector | None,
    ) -> tuple[MergeCandidate, ...]:
        discovered = tuple(self._graph.propose_candidates())
        scored = tuple(
            sorted(
                (self.candidate_score(candidate) for candidate in discovered),
                key=lambda item: (-item.total_score, item.candidate.candidate_sha256),
            )
        )
        if selector is None:
            ranked: tuple[Any, ...] = tuple(item.candidate for item in scored)
        else:
            selector_manifest = _tensor_identity_manifest(self.runtime)
            try:
                selected = self._invoke_selector(
                    selector,
                    Qwen3AttentionCandidateSelectionContext(
                        graph=self._graph,
                        discovered_candidates=tuple(item.candidate for item in scored),
                        scored_candidates=scored,
                        target_bytes=target_bytes,
                        current_bytes=current_bytes,
                        max_nonconflicting_batch_size=(
                            self.max_nonconflicting_batch_size
                        ),
                        evidence_epoch=self._evidence_epoch,
                    ),
                )
            finally:
                _assert_callback_tensor_contract(
                    self.runtime, selector_manifest, "candidate_selector"
                )
            if isinstance(selected, Mapping):
                forbidden = {
                    "role",
                    "roles",
                    "q",
                    "k",
                    "v",
                    "o",
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                }
                if forbidden & set(map(str, selected)):
                    raise Qwen3AttentionSurgeryShapeError(
                        "Candidate policies cannot request partial Q/K/V/O edits."
                    )
                ranked = tuple(selected.get("candidates", ())) + tuple(
                    selected.get("pairs", selected.get("explicit_pairs", ()))
                )
            elif isinstance(selected, (MergeCandidate, Qwen3AttentionCandidateScore)):
                ranked = (selected,)
            elif self._is_explicit_pair(selected):
                ranked = (selected,)
            elif isinstance(selected, Sequence) and not isinstance(
                selected, (str, bytes)
            ):
                ranked = tuple(selected)
            else:
                raise Qwen3AttentionSurgeryShapeError(
                    "candidate_selector must return candidates, scored candidates, "
                    "explicit pairs, or a mapping containing them."
                )

        selected_candidates: list[MergeCandidate] = []
        occupied: set[Qwen3AttentionNodeId] = set()
        explicit_count = 0
        for item in ranked:
            if isinstance(item, Qwen3AttentionCandidateScore):
                candidate = item.candidate
            elif isinstance(item, MergeCandidate):
                candidate = item
            elif self._is_explicit_pair(item):
                if explicit_count >= self.max_nonconflicting_batch_size:
                    break
                explicit_count += 1
                candidate = self._graph.propose_candidate_for_pair(
                    _node_ref(item[0]), _node_ref(item[1])
                )
            else:
                raise Qwen3AttentionSurgeryShapeError(
                    "Candidate selections must contain complete group-pair candidates."
                )
            if candidate.parent_snapshot_sha256 != self._graph.snapshot.snapshot_sha256:
                raise Qwen3AttentionSurgeryStateError(
                    "candidate_selector returned a stale candidate."
                )
            members = {_node_ref(node) for node in candidate.member_ids}
            if members & occupied:
                continue
            selected_candidates.append(candidate)
            occupied.update(members)
            if len(selected_candidates) >= self.max_nonconflicting_batch_size:
                break

        remaining = current_bytes - target_bytes
        batch: list[MergeCandidate] = []
        for candidate in selected_candidates:
            batch.append(candidate)
            if self._batch_physical_bytes_saved(batch) >= remaining:
                break
        while batch and self._batch_physical_bytes_saved(batch) <= 0:
            batch.pop()
        self._candidate_batches += 1
        return tuple(batch)

    def _rescore_batch(
        self, candidates: Sequence[MergeCandidate]
    ) -> tuple[MergeCandidate, ...]:
        return tuple(
            self._graph.propose_candidate_for_pair(candidate.sources)
            for candidate in candidates
        )

    def _invoke_refresh(
        self,
        callback: FeatureRefresh | None,
        context: Qwen3AttentionSurgeryRefreshContext,
        active_nodes: set[Qwen3AttentionNodeId],
    ) -> Qwen3AttentionEvidenceRefresh:
        retained_features = tuple(
            feature for feature in self._features if feature.node_id in active_nodes
        )
        retained_projections = tuple(
            projection
            for projection in self._projections
            if projection.source in active_nodes and projection.target in active_nodes
        )
        if callback is None:
            return Qwen3AttentionEvidenceRefresh(
                retained_features, retained_projections
            )
        callable_value = getattr(callback, "refresh", callback)
        if not callable(callable_value):
            raise Qwen3AttentionSurgeryShapeError(
                "refresh hook must be callable or expose refresh()."
            )
        try:
            signature = inspect.signature(callable_value)
        except (TypeError, ValueError):
            value = callable_value(context)
        else:
            positional = [
                parameter
                for parameter in signature.parameters.values()
                if parameter.kind
                in {
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                }
            ]
            if not positional:
                value = callable_value()
            elif len(positional) == 1:
                value = callable_value(context)
            else:
                value = callable_value(context.runtime, context)
        if isinstance(value, Qwen3AttentionEvidenceRefresh):
            refresh = value
        elif isinstance(value, Mapping):
            forbidden = {"q", "k", "v", "o", "roles"}
            if forbidden & set(map(str, value)):
                raise Qwen3AttentionSurgeryShapeError(
                    "Refresh hooks cannot return partial Q/K/V/O edits."
                )
            refresh = Qwen3AttentionEvidenceRefresh(
                value.get("features", ()),
                value.get("projections", retained_projections),
            )
        else:
            refresh = Qwen3AttentionEvidenceRefresh(value, retained_projections)
        self._feature_refreshes += 1
        return refresh

    def _validate_refresh(
        self,
        refresh: Qwen3AttentionEvidenceRefresh,
        active_nodes: set[Qwen3AttentionNodeId],
    ) -> None:
        new_by_node = self._feature_map(refresh.features)
        if (
            len(new_by_node) != len(refresh.features)
            or set(new_by_node) != active_nodes
        ):
            raise Qwen3AttentionSurgeryShapeError(
                "Accepted evidence refresh must return every active prototype exactly once."
            )
        old_by_node = self._feature_map(self._features)
        for node in active_nodes:
            old = old_by_node[node]
            new = new_by_node[node]
            if old.anchor != new.anchor or old.importance != new.importance:
                raise Qwen3AttentionSurgeryShapeError(
                    "Evidence refresh may change features only; anchors and importance "
                    "are immutable."
                )
        self._validate_projection_nodes(refresh.features, refresh.projections)

    def _assert_exact_anchors(
        self,
        before: Qwen3AttentionRuntimeSnapshot,
        after: Qwen3AttentionRuntimeSnapshot,
    ) -> None:
        before_hashes = dict(before.prototype_bundle_sha256s)
        after_hashes = dict(after.prototype_bundle_sha256s)
        for feature in self._features:
            if feature.anchor is None or feature.anchor.mode != "exact":
                continue
            node = feature.node_id
            if node not in after_hashes or after_hashes[node] != before_hashes[node]:
                raise Qwen3AttentionSurgeryStateError(
                    f"Exact attention anchor {node!r} was removed or changed."
                )

    def _attempt_batch(
        self,
        candidates: tuple[MergeCandidate, ...],
        checkpoint: _BridgeCheckpoint,
        *,
        target: int,
        gate: Gate | None,
        refresh_features: FeatureRefresh | None,
        max_attempts: int | None,
    ) -> bool:
        if max_attempts is not None and self._attempt_count >= max_attempts:
            raise Qwen3AttentionSurgeryTargetError(
                f"max_attempts={max_attempts} was reached at "
                f"{self.runtime_snapshot.tensor_bytes} bytes; target is {target}."
            )
        try:
            proposal = self._graph.preview(candidates)
            parent = _capture_installed(self.runtime)
            expected_savings = self._batch_physical_bytes_saved(candidates)
            if expected_savings <= 0:
                raise Qwen3AttentionSurgeryStateError(
                    "An attention transaction must have positive marginal physical bytes."
                )

            def projection_for_pair(source, target_node):
                return self._projection_for_pair_from(
                    source, target_node, self._projections
                )

            with torch.no_grad():
                tentative, used_projections = _compacted_attention_codebook(
                    self.runtime, proposal.snapshot, projection_for_pair
                )
                _install_attention_codebook(self.runtime, tentative)
            proposed = _capture_installed(self.runtime)
            measured_savings = (
                parent.snapshot.tensor_bytes - proposed.snapshot.tensor_bytes
            )
            if measured_savings != expected_savings or measured_savings <= 0:
                raise Qwen3AttentionSurgeryStateError(
                    "Predicted attention byte reward did not equal physical tensor storage."
                )
            self._assert_exact_anchors(parent.snapshot, proposed.snapshot)
            context = Qwen3AttentionSurgeryEvaluationContext(
                runtime=self.runtime,
                round_index=self._graph_rounds,
                evidence_epoch=self._evidence_epoch,
                target_bytes=target,
                parent_graph_snapshot=proposal.parent,
                proposed_graph_snapshot=proposal.snapshot,
                graph_transaction=proposal.transaction,
                parent_runtime_snapshot=parent.snapshot,
                proposed_runtime_snapshot=proposed.snapshot,
                projections=used_projections,
            )
            with torch.no_grad():
                gate_manifest = _tensor_identity_manifest(self.runtime)
                try:
                    decision, measured_claim = _invoke_gate(gate, context)
                finally:
                    _assert_callback_tensor_contract(self.runtime, gate_manifest, "gate")
            if (
                self.runtime_snapshot != proposed.snapshot
                or qwen3_codebook_parameter_report(self.runtime)
                != proposed.aggregate_report
            ):
                raise Qwen3AttentionSurgeryStateError(
                    "The gate attempted a partial or in-place Q/K/V/O tensor edit."
                )
            changes = _assignment_changes(parent.snapshot, proposed.snapshot)
            transaction_index = self._attempt_count + 1
            if not decision.accepted:
                if not _restore_installed(self.runtime, parent):
                    raise Qwen3AttentionSurgeryStateError(
                        "Rejected attention surgery did not restore exact tensors."
                    )
                outcome = self._graph.reject(proposal, decision)
                committed = self.runtime_snapshot
                self._attempt_count += 1
                self._rejected_count += 1
                self._append_transaction(
                    Qwen3AttentionSurgeryTransaction(
                        transaction_index=transaction_index,
                        round_index=self._graph_rounds,
                        evidence_epoch_before=self._evidence_epoch,
                        evidence_epoch_after=self._evidence_epoch,
                        accepted=False,
                        decision=decision,
                        graph_transaction_sha256=(
                            proposal.transaction.transaction_sha256
                        ),
                        action_sha256s=tuple(
                            action.action_sha256 for action in outcome.attempt.actions
                        ),
                        candidate_sha256s=tuple(
                            candidate.candidate_sha256 for candidate in candidates
                        ),
                        parent_runtime=Qwen3AttentionRuntimeAudit.from_snapshot(
                            parent.snapshot
                        ),
                        proposed_runtime=Qwen3AttentionRuntimeAudit.from_snapshot(
                            proposed.snapshot
                        ),
                        committed_runtime=Qwen3AttentionRuntimeAudit.from_snapshot(
                            committed
                        ),
                        assignment_change_count=len(changes),
                        assignment_changes_sha256=_sha256(changes),
                        projections=used_projections,
                        physical_bytes_saved=measured_savings,
                        rollback_exact=True,
                        measured_improvement_claim=measured_claim,
                    )
                )
                return False

            active_nodes = set(proposed.snapshot.active_node_ids)
            next_epoch = self._evidence_epoch + 1
            refresh_manifest = _tensor_identity_manifest(self.runtime)
            try:
                refresh = self._invoke_refresh(
                    refresh_features,
                    Qwen3AttentionSurgeryRefreshContext(
                        runtime=self.runtime,
                        round_index=self._graph_rounds,
                        accepted_transaction_index=self._accepted_count + 1,
                        evidence_epoch=next_epoch,
                        target_bytes=target,
                        current_bytes=proposed.snapshot.tensor_bytes,
                        active_node_ids=tuple(sorted(active_nodes)),
                        graph_transaction=proposal.transaction,
                    ),
                    active_nodes,
                )
            finally:
                _assert_callback_tensor_contract(
                    self.runtime, refresh_manifest, "refresh_features"
                )
            self._validate_refresh(refresh, active_nodes)
            next_graph = (
                self._build_graph(refresh.features, refresh.projections)
                if proposed.snapshot.tensor_bytes > target
                else None
            )
            outcome = self._graph.commit(proposal, decision)
            self._attempt_count += 1
            self._accepted_count += 1
            epoch_before = self._evidence_epoch
            self._evidence_epoch = next_epoch
            self._features = refresh.features
            self._projections = refresh.projections
            if next_graph is not None:
                self._graph = next_graph
                self._graph_rounds += 1
            self._append_transaction(
                Qwen3AttentionSurgeryTransaction(
                    transaction_index=transaction_index,
                    round_index=(
                        self._graph_rounds - 1
                        if next_graph is not None
                        else self._graph_rounds
                    ),
                    evidence_epoch_before=epoch_before,
                    evidence_epoch_after=self._evidence_epoch,
                    accepted=True,
                    decision=decision,
                    graph_transaction_sha256=outcome.transaction.transaction_sha256,
                    action_sha256s=tuple(
                        action.action_sha256 for action in outcome.transaction.actions
                    ),
                    candidate_sha256s=tuple(
                        candidate.candidate_sha256 for candidate in candidates
                    ),
                    parent_runtime=Qwen3AttentionRuntimeAudit.from_snapshot(
                        parent.snapshot
                    ),
                    proposed_runtime=Qwen3AttentionRuntimeAudit.from_snapshot(
                        proposed.snapshot
                    ),
                    committed_runtime=Qwen3AttentionRuntimeAudit.from_snapshot(
                        proposed.snapshot
                    ),
                    assignment_change_count=len(changes),
                    assignment_changes_sha256=_sha256(changes),
                    projections=used_projections,
                    physical_bytes_saved=measured_savings,
                    rollback_exact=None,
                    measured_improvement_claim=measured_claim,
                )
            )
            return True
        except Exception:
            self._restore_checkpoint(checkpoint)
            raise

    def run(
        self,
        target_bytes: int,
        *,
        gate: Gate | None = None,
        evaluator: Gate | None = None,
        refresh_features: FeatureRefresh | None = None,
        recompute_features: FeatureRefresh | None = None,
        max_attempts: int | None = None,
        candidate_selector: CandidateSelector | None = None,
    ) -> Qwen3AttentionSurgeryResult:
        """Fold complete attention groups until exact physical bytes meet target."""

        if self._has_run:
            raise Qwen3AttentionSurgeryStateError(
                "A Qwen3AttentionSurgeryBridge instance can run only once."
            )
        self._has_run = True
        if gate is not None and evaluator is not None:
            raise Qwen3AttentionSurgeryShapeError(
                "Provide gate or evaluator, not both."
            )
        rollout_gate = gate if gate is not None else evaluator
        if refresh_features is not None and recompute_features is not None:
            raise Qwen3AttentionSurgeryShapeError(
                "Provide refresh_features or recompute_features, not both."
            )
        refresh = (
            refresh_features if refresh_features is not None else recompute_features
        )
        selector = (
            candidate_selector
            if candidate_selector is not None
            else self._candidate_selector
        )
        if selector is not None and not callable(getattr(selector, "select", selector)):
            raise Qwen3AttentionSurgeryShapeError(
                "candidate_selector must be callable or expose select()."
            )
        target = _nonnegative_int(target_bytes, "target_bytes")
        attempt_limit = (
            None
            if max_attempts is None
            else _positive_int(max_attempts, "max_attempts")
        )
        initial = self.runtime_snapshot
        current = initial
        if target >= current.tensor_bytes:
            return Qwen3AttentionSurgeryResult(
                runtime=self.runtime,
                target_bytes=target,
                initial_runtime_snapshot=initial,
                final_runtime_snapshot=current,
                transactions=(),
                diagnostics=self.diagnostics,
                target_reached=True,
            )

        def resolve_batch(
            candidates: tuple[MergeCandidate, ...],
            checkpoint: _BridgeCheckpoint,
        ) -> None:
            nonlocal current
            if self._batch_physical_bytes_saved(candidates) <= 0:
                return
            accepted = self._attempt_batch(
                candidates,
                checkpoint,
                target=target,
                gate=rollout_gate,
                refresh_features=refresh,
                max_attempts=attempt_limit,
            )
            current = self.runtime_snapshot
            if accepted or len(candidates) == 1:
                return
            self._bisections += 1
            midpoint = len(candidates) // 2
            for part in (candidates[:midpoint], candidates[midpoint:]):
                if current.tensor_bytes <= target:
                    return
                part_checkpoint = self._checkpoint()
                try:
                    rescored = self._rescore_batch(part)
                    resolve_batch(rescored, part_checkpoint)
                    current = self.runtime_snapshot
                except Exception:
                    self._restore_checkpoint(part_checkpoint)
                    raise

        try:
            while current.tensor_bytes > target:
                if attempt_limit is not None and self._attempt_count >= attempt_limit:
                    raise Qwen3AttentionSurgeryTargetError(
                        f"max_attempts={attempt_limit} was reached at "
                        f"{current.tensor_bytes} bytes; target is {target}."
                    )
                selection_checkpoint = self._checkpoint()
                try:
                    candidates = self._selected_candidates(
                        target_bytes=target,
                        current_bytes=current.tensor_bytes,
                        selector=selector,
                    )
                except SurgerySearchExhausted as error:
                    self._restore_checkpoint(selection_checkpoint)
                    raise Qwen3AttentionSurgeryTargetError(str(error)) from error
                except Exception:
                    self._restore_checkpoint(selection_checkpoint)
                    raise
                if not candidates:
                    self._restore_checkpoint(selection_checkpoint)
                    raise Qwen3AttentionSurgeryTargetError(
                        "No positive-byte coupled attention batch remains at "
                        f"{current.tensor_bytes} bytes; target is {target}."
                    )
                resolve_batch(candidates, selection_checkpoint)
                current = self.runtime_snapshot
        except Exception:
            self._has_run = False
            raise

        final = self.runtime_snapshot
        return Qwen3AttentionSurgeryResult(
            runtime=self.runtime,
            target_bytes=target,
            initial_runtime_snapshot=initial,
            final_runtime_snapshot=final,
            transactions=tuple(self._transactions),
            diagnostics=self.diagnostics,
            target_reached=final.tensor_bytes <= target,
        )


def apply_qwen3_attention_surgery(
    runtime: nn.Module,
    features: Sequence[Qwen3AttentionSurgeryFeature],
    *,
    target_bytes: int,
    gate: Gate | None = None,
    evaluator: Gate | None = None,
    refresh_features: FeatureRefresh | None = None,
    recompute_features: FeatureRefresh | None = None,
    projections: Sequence[Qwen3AttentionProjection] | Mapping[Any, Any] = (),
    feature_bounds: Qwen3AttentionFeatureBounds | None = None,
    score_weights: ScoreWeights | None = None,
    projection_count: int = 8,
    neighbor_window: int = 4,
    max_candidates: int = 64,
    medoid_candidate_limit: int = 32,
    max_nonconflicting_batch_size: int = 1,
    transaction_history_limit: int = 32,
    max_attempts: int | None = None,
    candidate_selector: CandidateSelector | None = None,
) -> Qwen3AttentionSurgeryResult:
    bridge = Qwen3AttentionSurgeryBridge(
        runtime,
        features,
        projections=projections,
        feature_bounds=feature_bounds,
        score_weights=score_weights,
        projection_count=projection_count,
        neighbor_window=neighbor_window,
        max_candidates=max_candidates,
        medoid_candidate_limit=medoid_candidate_limit,
        max_nonconflicting_batch_size=max_nonconflicting_batch_size,
        transaction_history_limit=transaction_history_limit,
        candidate_selector=candidate_selector,
    )
    return bridge.run(
        target_bytes,
        gate=gate,
        evaluator=evaluator,
        refresh_features=refresh_features,
        recompute_features=recompute_features,
        max_attempts=max_attempts,
    )


Qwen3GQAAttentionSurgeryBridge = Qwen3AttentionSurgeryBridge
