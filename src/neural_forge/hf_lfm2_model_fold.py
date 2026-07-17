"""Atomic zero-training depth-and-width folding for Transformers LFM2 models.

The layer primitive in :mod:`neural_forge.hf_lfm2_closed_loop_fold` already
implements coupled GQA and SwiGLU folding.  This module composes those immutable
layer candidates into one *model-level* transaction and adds an honest depth
fold:

* source layers are partitioned into ordered, contiguous spans;
* every target layer owns exactly one source span;
* singleton spans use the existing closed-loop layer fold unchanged;
* a multi-layer span chooses one real, topology-compatible source layer as a
  deterministic anchor;
* the anchor's child-side MLP design is fitted to the cached residual at the
  end of the complete source span, with bounded rank and norm correction; and
* every target state tensor, including embeddings, final norm, and output head,
  is overwritten in one atomic installation.

A multi-layer span is not represented as an exact algebraic composition.  Its
report explicitly identifies the approximation as a real-layer anchor plus a
bounded residual-endpoint correction.  This module never calls either model,
collects traces, constructs an optimizer, calls backward, or writes gradients.
All behavioral evidence must already have been materialized from a verified
cache by the caller.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from types import MappingProxyType
from typing import Any

from neural_forge.hf_activation_cluster import (
    build_gated_mlp_activation_cluster_plan,
    build_gqa_activation_cluster_plan,
)
from neural_forge.hf_lfm2_canonical import (
    Lfm2AttentionContract,
    Lfm2LayerContract,
    inspect_lfm2_layer,
)
from neural_forge.hf_lfm2_closed_loop_fold import (
    Lfm2AttentionFoldEvidence,
    Lfm2ClosedLoopFoldConfig,
    Lfm2ClosedLoopFoldEvidence,
    Lfm2ClosedLoopFoldPlan,
    Lfm2MLPFoldEvidence,
    build_lfm2_closed_loop_fold_plan,
)


MODEL_FOLD_METHOD = "lfm2_hybrid_residual_attention_model"
MODEL_FOLD_SCHEMA = "neural-forge-lfm2-hybrid-model-fold-v1"
MODEL_FOLD_TRANSACTION_SCHEMA = (
    "neural-forge-lfm2-hybrid-model-fold-transaction-v1"
)
MODEL_FOLD_EVIDENCE_SCHEMA = "neural-forge-lfm2-hybrid-model-evidence-v1"


class Lfm2ModelFoldError(ValueError):
    """Base failure for an LFM2 model-level fold."""


class Lfm2ModelFoldTopologyError(Lfm2ModelFoldError):
    """The source spans or target model do not form a supported fold."""


class Lfm2ModelFoldEvidenceError(Lfm2ModelFoldError):
    """Cached span evidence is incomplete, unaligned, or non-finite."""


class Lfm2ModelFoldInvariantError(RuntimeError):
    """An atomicity, provenance, or zero-training invariant failed."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError("PyTorch is required for LFM2 model folding.") from exc
    return torch


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise Lfm2ModelFoldError(f"{name} must be a non-negative integer.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2ModelFoldError(
            f"{name} must be a non-negative integer."
        ) from exc
    if result < 0:
        raise Lfm2ModelFoldError(f"{name} must be a non-negative integer.")
    return int(result)


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise Lfm2ModelFoldError(f"{name} must be a positive integer.")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2ModelFoldError(f"{name} must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2ModelFoldError(f"{name} must be finite and non-negative.") from exc
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2ModelFoldError(f"{name} must be finite and non-negative.")
    return result


def _sha256_digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2ModelFoldEvidenceError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Lfm2ModelFoldInvariantError(
            "Model-fold provenance is not canonical JSON."
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise Lfm2ModelFoldInvariantError("Cannot fingerprint a non-tensor value.")
    if value.layout != torch.strided or value.device.type == "meta":
        raise Lfm2ModelFoldInvariantError(
            "Model-fold tensors must be materialized and dense."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(b"neural-forge-lfm2-model-fold-tensor-v1\0")
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            [int(size) for size in tensor.shape],
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).reshape(-1).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _clone_tensor(value: Any, name: str) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise Lfm2ModelFoldInvariantError(f"{name} must be a tensor.")
    if value.layout != torch.strided or value.device.type == "meta":
        raise Lfm2ModelFoldInvariantError(
            f"{name} must be a materialized dense tensor."
        )
    result = value.detach().to(device="cpu").contiguous().clone()
    if (result.is_floating_point() or result.is_complex()) and not bool(
        torch.isfinite(result).all().item()
    ):
        raise Lfm2ModelFoldInvariantError(f"{name} contains NaN or infinity.")
    return result


def _float64_evidence_tensor(value: Any, name: str, *, ndim: int) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != ndim
        or any(int(size) <= 0 for size in value.shape)
        or not value.is_floating_point()
        or value.is_complex()
    ):
        raise Lfm2ModelFoldEvidenceError(
            f"{name} must be a non-empty real floating rank-{ndim} tensor."
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2ModelFoldEvidenceError(f"{name} contains NaN or infinity.")
    return result


def _float64_rows(value: Any, name: str) -> Any:
    return _float64_evidence_tensor(value, name, ndim=2)


def _rows_equal(left: Any, right: Any) -> bool:
    return bool(left.shape == right.shape and _torch().equal(left, right))


def _state_items(module: Any) -> tuple[tuple[str, Any], ...]:
    try:
        state = module.state_dict()
    except Exception as exc:
        raise Lfm2ModelFoldInvariantError("Could not read model state.") from exc
    if not isinstance(state, Mapping) or not state:
        raise Lfm2ModelFoldInvariantError("An LFM2 model cannot have empty state.")
    result = []
    for name in sorted(state):
        if not isinstance(name, str):
            raise Lfm2ModelFoldInvariantError(
                "Model state names must be strings."
            )
        result.append((name, _clone_tensor(state[name], f"state[{name!r}]")))
    return tuple(result)


def _state_mapping(
    items: tuple[tuple[str, Any], ...],
) -> Mapping[str, Any]:
    return MappingProxyType({name: value for name, value in items})


def _state_spec(
    values: Mapping[str, Any],
) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    return tuple(
        (
            name,
            tuple(int(size) for size in values[name].shape),
            str(values[name].dtype),
        )
        for name in sorted(values)
    )


def _state_sha256(values: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(b"neural-forge-lfm2-model-state-v1\0")
    for name in sorted(values):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_tensor_sha256(values[name])))
    return digest.hexdigest()


def _live_state(module: Any) -> Mapping[str, Any]:
    state = module.state_dict()
    if not isinstance(state, Mapping) or not state:
        raise Lfm2ModelFoldInvariantError("An LFM2 model cannot have empty state.")
    return state


def _live_state_sha256(module: Any) -> str:
    return _state_sha256(_live_state(module))


def _named_parameters(module: Any) -> tuple[tuple[str, Any], ...]:
    try:
        return tuple(module.named_parameters(remove_duplicate=False))
    except TypeError:  # pragma: no cover - old optional torch
        return tuple(module.named_parameters())


def _named_buffers(module: Any) -> tuple[tuple[str, Any], ...]:
    try:
        return tuple(module.named_buffers(remove_duplicate=False))
    except TypeError:  # pragma: no cover - old optional torch
        return tuple(module.named_buffers())


def _parameter_ids(module: Any) -> tuple[tuple[str, int], ...]:
    return tuple((name, id(parameter)) for name, parameter in _named_parameters(module))


def _gradient_signature(
    module: Any,
) -> tuple[tuple[str, int, bool, int | None, str | None], ...]:
    result = []
    for name, parameter in _named_parameters(module):
        gradient = parameter.grad
        result.append(
            (
                name,
                id(parameter),
                bool(parameter.requires_grad),
                None if gradient is None else id(gradient),
                None if gradient is None else _tensor_sha256(gradient),
            )
        )
    return tuple(result)


def _buffer_identity(
    module: Any,
) -> tuple[tuple[str, int, tuple[int, ...], str], ...]:
    return tuple(
        (
            name,
            id(value),
            tuple(int(size) for size in value.shape),
            str(value.dtype),
        )
        for name, value in _named_buffers(module)
    )


def _buffer_items(module: Any) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (name, _clone_tensor(value, f"buffer[{name!r}]"))
        for name, value in _named_buffers(module)
    )


def _module_mode_signature(module: Any) -> tuple[tuple[str, int, bool], ...]:
    return tuple(
        (name, id(value), bool(value.training))
        for name, value in module.named_modules()
    )


def _model_config_sha256(module: Any) -> str | None:
    config = getattr(module, "config", None)
    to_dict = getattr(config, "to_dict", None)
    if not callable(to_dict):
        return None
    try:
        value = to_dict()
    except Exception as exc:
        raise Lfm2ModelFoldInvariantError(
            "Could not serialize the model config."
        ) from exc
    if not isinstance(value, Mapping):
        raise Lfm2ModelFoldInvariantError("config.to_dict() must return a mapping.")
    return _canonical_sha256(dict(value))


def _unique_parameter_count(module: Any) -> int:
    seen: set[int] = set()
    total = 0
    for _name, parameter in _named_parameters(module):
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        total += int(parameter.numel())
    if total <= 0:
        raise Lfm2ModelFoldInvariantError(
            "An LFM2 model must expose learned parameters."
        )
    return total


def _evidence_identity(value: Any) -> Any:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        tensor = _clone_tensor(value, "evidence tensor")
        return {
            "kind": "tensor",
            "dtype": str(tensor.dtype),
            "shape": [int(size) for size in tensor.shape],
            "sha256": _tensor_sha256(tensor),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _evidence_identity(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _evidence_identity(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_evidence_identity(item) for item in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    raise Lfm2ModelFoldEvidenceError(
        f"Unsupported cached evidence value type: {type(value)!r}."
    )


@dataclass(frozen=True, slots=True)
class Lfm2ModelFoldSpan:
    """One ordered source-layer span assigned to one target layer."""

    source_layer_indices: tuple[int, ...]
    target_layer_index: int

    def __post_init__(self) -> None:
        try:
            source = tuple(
                _nonnegative_int(value, "source layer index")
                for value in self.source_layer_indices
            )
        except TypeError as exc:
            raise Lfm2ModelFoldTopologyError(
                "source_layer_indices must be an integer sequence."
            ) from exc
        if not source:
            raise Lfm2ModelFoldTopologyError(
                "A model-fold span must contain a source layer."
            )
        if source != tuple(range(source[0], source[-1] + 1)):
            raise Lfm2ModelFoldTopologyError(
                "Every model-fold source span must be contiguous and ordered."
            )
        object.__setattr__(self, "source_layer_indices", source)
        object.__setattr__(
            self,
            "target_layer_index",
            _nonnegative_int(self.target_layer_index, "target_layer_index"),
        )

    @property
    def is_depth_fold(self) -> bool:
        return len(self.source_layer_indices) > 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_layer_indices": list(self.source_layer_indices),
            "target_layer_index": self.target_layer_index,
            "source_layer_count": len(self.source_layer_indices),
            "depth_fold": self.is_depth_fold,
        }


def build_leading_pair_lfm2_span_plan(
    source_layer_count: int = 14,
) -> tuple[Lfm2ModelFoldSpan, ...]:
    """Merge source layers 0/1, then map every remaining layer one-to-one."""

    count = _positive_int(source_layer_count, "source_layer_count")
    if count < 2:
        raise Lfm2ModelFoldTopologyError(
            "A leading-pair span plan requires at least two source layers."
        )
    return (
        Lfm2ModelFoldSpan((0, 1), 0),
        *(
            Lfm2ModelFoldSpan((source_index,), source_index - 1)
            for source_index in range(2, count)
        ),
    )


@dataclass(frozen=True, slots=True)
class Lfm2CachedLayerFoldEvidence:
    """Existing layer-fold evidence plus its cached residual endpoints."""

    source_layer_index: int
    fold_evidence: Lfm2ClosedLoopFoldEvidence
    parent_pre_residual_rows: Any
    parent_post_residual_rows: Any
    validation_parent_pre_residual_rows: Any | None = None
    validation_parent_post_residual_rows: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_layer_index",
            _nonnegative_int(self.source_layer_index, "source_layer_index"),
        )
        if not isinstance(self.fold_evidence, Lfm2ClosedLoopFoldEvidence):
            raise Lfm2ModelFoldEvidenceError(
                "fold_evidence must be Lfm2ClosedLoopFoldEvidence."
            )
        validation = (
            self.validation_parent_pre_residual_rows is not None,
            self.validation_parent_post_residual_rows is not None,
        )
        if any(validation) and not all(validation):
            raise Lfm2ModelFoldEvidenceError(
                "Layer validation pre/post residual rows must be supplied together."
            )


@dataclass(frozen=True, slots=True)
class Lfm2CachedSpanFoldEvidence:
    """Cache-only evidence for one complete source span."""

    span: Lfm2ModelFoldSpan
    layers: tuple[Lfm2CachedLayerFoldEvidence, ...]
    parent_pre_span_residual_rows: Any
    parent_post_span_residual_rows: Any
    validation_parent_pre_span_residual_rows: Any | None = None
    validation_parent_post_span_residual_rows: Any | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.span, Lfm2ModelFoldSpan):
            raise Lfm2ModelFoldEvidenceError("span must be Lfm2ModelFoldSpan.")
        try:
            layer_values = tuple(self.layers)
        except TypeError as exc:
            raise Lfm2ModelFoldEvidenceError(
                "layers must be a cached layer evidence sequence."
            ) from exc
        if any(
            not isinstance(value, Lfm2CachedLayerFoldEvidence)
            for value in layer_values
        ):
            raise Lfm2ModelFoldEvidenceError(
                "layers must contain Lfm2CachedLayerFoldEvidence values."
            )
        observed = tuple(value.source_layer_index for value in layer_values)
        if observed != self.span.source_layer_indices:
            raise Lfm2ModelFoldEvidenceError(
                "Cached layer evidence does not exactly cover its declared span."
            )
        object.__setattr__(self, "layers", layer_values)
        validation = (
            self.validation_parent_pre_span_residual_rows is not None,
            self.validation_parent_post_span_residual_rows is not None,
        )
        if any(validation) and not all(validation):
            raise Lfm2ModelFoldEvidenceError(
                "Span validation pre/post residual rows must be supplied together."
            )


@dataclass(frozen=True, slots=True)
class Lfm2ModelFoldEvidence:
    """One immutable, cache-bound evidence bundle for the whole model fold."""

    cache_fingerprint: str
    child_feature_state_sha256: str
    spans: tuple[Lfm2CachedSpanFoldEvidence, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cache_fingerprint",
            _sha256_digest(self.cache_fingerprint, "cache_fingerprint"),
        )
        object.__setattr__(
            self,
            "child_feature_state_sha256",
            _sha256_digest(
                self.child_feature_state_sha256,
                "child_feature_state_sha256",
            ),
        )
        try:
            span_values = tuple(self.spans)
        except TypeError as exc:
            raise Lfm2ModelFoldEvidenceError(
                "spans must be a cached span evidence sequence."
            ) from exc
        if not span_values or any(
            not isinstance(value, Lfm2CachedSpanFoldEvidence)
            for value in span_values
        ):
            raise Lfm2ModelFoldEvidenceError(
                "spans must contain cached span evidence."
            )
        targets = tuple(value.span.target_layer_index for value in span_values)
        if len(set(targets)) != len(targets):
            raise Lfm2ModelFoldEvidenceError(
                "Cached span evidence repeats a target layer."
            )
        object.__setattr__(self, "spans", span_values)


def fingerprint_lfm2_model_fold_evidence(
    evidence: Lfm2ModelFoldEvidence,
) -> str:
    """Hash every cached tensor and structural evidence field."""

    if not isinstance(evidence, Lfm2ModelFoldEvidence):
        raise TypeError("evidence must be Lfm2ModelFoldEvidence.")
    return _canonical_sha256(
        {
            "schema": MODEL_FOLD_EVIDENCE_SCHEMA,
            "cached_evidence_only": True,
            "parent_forward_count": 0,
            "payload": _evidence_identity(evidence),
        }
    )


def fingerprint_lfm2_model_feature_state(model: Any) -> str:
    """Fingerprint the exact child state used to capture child-side designs."""

    _decoder_layers(model)
    return _live_state_sha256(model)


def assemble_lfm2_model_fold_evidence(
    *,
    cache_fingerprint: str,
    child_feature_state_sha256: str,
    spans: Sequence[Lfm2ModelFoldSpan],
    layer_fold_evidence: Mapping[int, Lfm2ClosedLoopFoldEvidence],
    parent_pre_residual_rows: Mapping[int, Any],
    parent_post_residual_rows: Mapping[int, Any],
    validation_parent_pre_residual_rows: Mapping[int, Any] | None = None,
    validation_parent_post_residual_rows: Mapping[int, Any] | None = None,
) -> Lfm2ModelFoldEvidence:
    """Assemble model evidence from already-materialized verified cache rows.

    No model is accepted by this helper, so it cannot regenerate parent traces
    or derive child designs.  ``layer_fold_evidence`` must already contain the
    child-side target designs required by the existing layer primitive.
    """

    selected = tuple(spans)
    validation_supplied = (
        validation_parent_pre_residual_rows is not None,
        validation_parent_post_residual_rows is not None,
    )
    if any(validation_supplied) and not all(validation_supplied):
        raise Lfm2ModelFoldEvidenceError(
            "Validation residual mappings must be supplied together."
        )
    expected_indices = {
        index for span in selected for index in span.source_layer_indices
    }
    for name, values in (
        ("layer_fold_evidence", layer_fold_evidence),
        ("parent_pre_residual_rows", parent_pre_residual_rows),
        ("parent_post_residual_rows", parent_post_residual_rows),
    ):
        if set(values) != expected_indices:
            raise Lfm2ModelFoldEvidenceError(
                f"{name} does not exactly cover the declared source layers."
            )
    if validation_parent_pre_residual_rows is not None:
        assert validation_parent_post_residual_rows is not None
        if (
            set(validation_parent_pre_residual_rows) != expected_indices
            or set(validation_parent_post_residual_rows) != expected_indices
        ):
            raise Lfm2ModelFoldEvidenceError(
                "Validation residual mappings do not exactly cover source layers."
            )

    cached_spans = []
    for span in selected:
        cached_layers = tuple(
            Lfm2CachedLayerFoldEvidence(
                source_layer_index=index,
                fold_evidence=layer_fold_evidence[index],
                parent_pre_residual_rows=parent_pre_residual_rows[index],
                parent_post_residual_rows=parent_post_residual_rows[index],
                validation_parent_pre_residual_rows=(
                    None
                    if validation_parent_pre_residual_rows is None
                    else validation_parent_pre_residual_rows[index]
                ),
                validation_parent_post_residual_rows=(
                    None
                    if validation_parent_post_residual_rows is None
                    else validation_parent_post_residual_rows[index]
                ),
            )
            for index in span.source_layer_indices
        )
        cached_spans.append(
            Lfm2CachedSpanFoldEvidence(
                span=span,
                layers=cached_layers,
                parent_pre_span_residual_rows=(
                    parent_pre_residual_rows[span.source_layer_indices[0]]
                ),
                parent_post_span_residual_rows=(
                    parent_post_residual_rows[span.source_layer_indices[-1]]
                ),
                validation_parent_pre_span_residual_rows=(
                    None
                    if validation_parent_pre_residual_rows is None
                    else validation_parent_pre_residual_rows[
                        span.source_layer_indices[0]
                    ]
                ),
                validation_parent_post_span_residual_rows=(
                    None
                    if validation_parent_post_residual_rows is None
                    else validation_parent_post_residual_rows[
                        span.source_layer_indices[-1]
                    ]
                ),
            )
        )
    return Lfm2ModelFoldEvidence(
        cache_fingerprint=cache_fingerprint,
        child_feature_state_sha256=child_feature_state_sha256,
        spans=tuple(cached_spans),
    )


def _trace_rows_metadata(
    rows: Any,
    name: str,
) -> tuple[tuple[str, ...], tuple[str, ...], str, tuple[str, ...]]:
    paths = tuple(str(value) for value in getattr(rows, "paths", ()))
    splits = tuple(str(value) for value in getattr(rows, "path_splits", ()))
    corpus = getattr(rows, "corpus_fingerprint", None)
    kinds = tuple(str(value) for value in getattr(rows, "layer_kinds", ()))
    if (
        not paths
        or len(paths) != len(splits)
        or not isinstance(corpus, str)
        or len(corpus) != 64
        or any(character not in "0123456789abcdef" for character in corpus)
        or not kinds
        or not callable(getattr(rows, "field", None))
    ):
        raise Lfm2ModelFoldEvidenceError(
            f"{name} must expose aligned paths/splits, a cache corpus digest, "
            "ordered layer kinds, and field()."
        )
    return paths, splits, corpus, kinds


def _validated_trace_indices(
    values: Sequence[int],
    *,
    row_count: int,
    name: str,
    allow_empty: bool,
) -> tuple[int, ...]:
    try:
        result = tuple(
            _nonnegative_int(value, f"{name} row index")
            for value in values
        )
    except TypeError as exc:
        raise Lfm2ModelFoldEvidenceError(
            f"{name} must be an integer row-index sequence."
        ) from exc
    if (not result and not allow_empty) or len(set(result)) != len(result):
        qualifier = "possibly empty unique" if allow_empty else "nonempty unique"
        raise Lfm2ModelFoldEvidenceError(
            f"{name} must contain {qualifier} row indices."
        )
    if any(value >= row_count for value in result):
        raise Lfm2ModelFoldEvidenceError(
            f"{name} contains an out-of-range row index."
        )
    return result


def _index_trace_rows(value: Any, indices: tuple[int, ...], name: str) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim < 1
        or int(value.shape[0]) <= 0
    ):
        raise Lfm2ModelFoldEvidenceError(
            f"{name} must be a materialized dense tensor with a row axis."
        )
    index = torch.tensor(indices, dtype=torch.long, device=value.device)
    result = value.index_select(0, index).detach().to(device="cpu").contiguous()
    if (result.is_floating_point() or result.is_complex()) and not bool(
        torch.isfinite(result).all().item()
    ):
        raise Lfm2ModelFoldEvidenceError(f"{name} contains NaN or infinity.")
    return result


def assemble_lfm2_model_fold_evidence_from_trace_rows(
    *,
    cache_fingerprint: str,
    child_feature_state_sha256: str,
    spans: Sequence[Lfm2ModelFoldSpan],
    parent_rows: Any,
    child_rows: Any,
    selection_indices: Sequence[int],
    fit_indices: Sequence[int],
    validation_indices: Sequence[int] = (),
) -> Lfm2ModelFoldEvidence:
    """Assemble hybrid evidence from one sealed parent and producer-child capture.

    This is the model-free extraction previously embedded in the production
    catalog callback.  It accepts only already-materialized trace rows, so the
    hybrid transaction can install a complete producer shell first, capture
    that shell exactly once, and then build the final model plan without ever
    asking the evidence provider to execute the compiler-zero child.
    """

    try:
        span_values = tuple(spans)
    except TypeError as exc:
        raise Lfm2ModelFoldEvidenceError(
            "spans must be an LFM2 model-fold span sequence."
        ) from exc
    if not span_values or any(
        not isinstance(value, Lfm2ModelFoldSpan) for value in span_values
    ):
        raise Lfm2ModelFoldEvidenceError(
            "spans must contain Lfm2ModelFoldSpan values."
        )
    parent_paths, parent_splits, parent_corpus, parent_kinds = (
        _trace_rows_metadata(parent_rows, "parent_rows")
    )
    child_paths, child_splits, child_corpus, child_kinds = _trace_rows_metadata(
        child_rows,
        "child_rows",
    )
    if (
        child_paths != parent_paths
        or child_splits != parent_splits
        or child_corpus != parent_corpus
    ):
        raise Lfm2ModelFoldEvidenceError(
            "Parent and producer-child trace paths are not exactly aligned."
        )
    _validate_span_partition(
        span_values,
        len(parent_kinds),
        len(child_kinds),
    )
    selection = _validated_trace_indices(
        selection_indices,
        row_count=len(parent_paths),
        name="selection_indices",
        allow_empty=False,
    )
    fit = _validated_trace_indices(
        fit_indices,
        row_count=len(parent_paths),
        name="fit_indices",
        allow_empty=False,
    )
    validation = _validated_trace_indices(
        validation_indices,
        row_count=len(parent_paths),
        name="validation_indices",
        allow_empty=True,
    )
    if (
        set(selection) & set(fit)
        or set(selection) & set(validation)
        or set(fit) & set(validation)
    ):
        raise Lfm2ModelFoldEvidenceError(
            "Selection, fit, and validation rows must be disjoint."
        )

    source_to_target = {
        source_index: span.target_layer_index
        for span in span_values
        for source_index in span.source_layer_indices
    }
    layer_evidence: dict[int, Lfm2ClosedLoopFoldEvidence] = {}
    parent_pre: dict[int, Any] = {}
    parent_post: dict[int, Any] = {}
    validation_pre: dict[int, Any] = {}
    validation_post: dict[int, Any] = {}
    for source_index, target_index in sorted(source_to_target.items()):
        prefix = f"source layer {source_index}"
        parent_pre[source_index] = _index_trace_rows(
            parent_rows.field(source_index, "residual.pre"),
            fit,
            f"{prefix} residual.pre",
        )
        parent_post[source_index] = _index_trace_rows(
            parent_rows.field(source_index, "residual.post_layer"),
            fit,
            f"{prefix} residual.post_layer",
        )
        if validation:
            validation_pre[source_index] = _index_trace_rows(
                parent_rows.field(source_index, "residual.pre"),
                validation,
                f"{prefix} validation residual.pre",
            )
            validation_post[source_index] = _index_trace_rows(
                parent_rows.field(source_index, "residual.post_layer"),
                validation,
                f"{prefix} validation residual.post_layer",
            )
        mlp = Lfm2MLPFoldEvidence(
            parent_gate_rows=_index_trace_rows(
                parent_rows.field(source_index, "mlp.gate"),
                selection,
                f"{prefix} mlp.gate",
            ),
            parent_up_rows=_index_trace_rows(
                parent_rows.field(source_index, "mlp.up"),
                selection,
                f"{prefix} mlp.up",
            ),
            parent_post_swiglu_rows=_index_trace_rows(
                parent_rows.field(source_index, "mlp.post_swiglu"),
                selection,
                f"{prefix} mlp.post_swiglu",
            ),
            parent_contribution_rows=None,
            child_post_swiglu_rows=_index_trace_rows(
                child_rows.field(target_index, "mlp.post_swiglu"),
                fit,
                f"target layer {target_index} mlp.post_swiglu",
            ),
            parent_branch_output_rows=parent_post[source_index],
            child_pre_branch_residual_rows=_index_trace_rows(
                child_rows.field(target_index, "residual.post_operator"),
                fit,
                f"target layer {target_index} residual.post_operator",
            ),
            validation_child_post_swiglu_rows=(
                None
                if not validation
                else _index_trace_rows(
                    child_rows.field(target_index, "mlp.post_swiglu"),
                    validation,
                    f"target layer {target_index} validation mlp.post_swiglu",
                )
            ),
            validation_parent_branch_output_rows=(
                None if not validation else validation_post[source_index]
            ),
            validation_child_pre_branch_residual_rows=(
                None
                if not validation
                else _index_trace_rows(
                    child_rows.field(target_index, "residual.post_operator"),
                    validation,
                    (
                        f"target layer {target_index} validation "
                        "residual.post_operator"
                    ),
                )
            ),
        )
        attention = None
        if (
            parent_kinds[source_index] == "full_attention"
            and child_kinds[target_index] == "full_attention"
        ):
            attention = Lfm2AttentionFoldEvidence(
                parent_query_rows=_index_trace_rows(
                    parent_rows.field(source_index, "attention.q"),
                    selection,
                    f"{prefix} attention.q",
                ),
                parent_key_rows=_index_trace_rows(
                    parent_rows.field(source_index, "attention.k"),
                    selection,
                    f"{prefix} attention.k",
                ),
                parent_value_rows=_index_trace_rows(
                    parent_rows.field(source_index, "attention.v"),
                    selection,
                    f"{prefix} attention.v",
                ),
                parent_context_rows=_index_trace_rows(
                    parent_rows.field(source_index, "attention.context"),
                    selection,
                    f"{prefix} attention.context",
                ),
                parent_contribution_rows=None,
                child_context_rows=_index_trace_rows(
                    child_rows.field(target_index, "attention.context"),
                    fit,
                    f"target layer {target_index} attention.context",
                ),
                parent_branch_output_rows=_index_trace_rows(
                    parent_rows.field(source_index, "residual.post_operator"),
                    fit,
                    f"{prefix} residual.post_operator",
                ),
                child_pre_branch_residual_rows=_index_trace_rows(
                    child_rows.field(target_index, "residual.pre"),
                    fit,
                    f"target layer {target_index} residual.pre",
                ),
                validation_child_context_rows=(
                    None
                    if not validation
                    else _index_trace_rows(
                        child_rows.field(target_index, "attention.context"),
                        validation,
                        (
                            f"target layer {target_index} validation "
                            "attention.context"
                        ),
                    )
                ),
                validation_parent_branch_output_rows=(
                    None
                    if not validation
                    else _index_trace_rows(
                        parent_rows.field(
                            source_index,
                            "residual.post_operator",
                        ),
                        validation,
                        f"{prefix} validation residual.post_operator",
                    )
                ),
                validation_child_pre_branch_residual_rows=(
                    None
                    if not validation
                    else _index_trace_rows(
                        child_rows.field(target_index, "residual.pre"),
                        validation,
                        f"target layer {target_index} validation residual.pre",
                    )
                ),
            )
        layer_evidence[source_index] = Lfm2ClosedLoopFoldEvidence(
            mlp=mlp,
            attention=attention,
        )
    return assemble_lfm2_model_fold_evidence(
        cache_fingerprint=cache_fingerprint,
        child_feature_state_sha256=child_feature_state_sha256,
        spans=span_values,
        layer_fold_evidence=layer_evidence,
        parent_pre_residual_rows=parent_pre,
        parent_post_residual_rows=parent_post,
        validation_parent_pre_residual_rows=(
            validation_pre if validation else None
        ),
        validation_parent_post_residual_rows=(
            validation_post if validation else None
        ),
    )


@dataclass(frozen=True, slots=True)
class Lfm2ModelFoldConfig:
    """Layer fit policy plus hard bounds for multi-layer endpoint correction."""

    layer_config: Lfm2ClosedLoopFoldConfig = field(
        default_factory=Lfm2ClosedLoopFoldConfig
    )
    max_span_correction_anchors: int = 128
    max_span_endpoint_prior_delta_ratio: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.layer_config, Lfm2ClosedLoopFoldConfig):
            raise TypeError(
                "layer_config must be Lfm2ClosedLoopFoldConfig."
            )
        object.__setattr__(
            self,
            "max_span_correction_anchors",
            _positive_int(
                self.max_span_correction_anchors,
                "max_span_correction_anchors",
            ),
        )
        object.__setattr__(
            self,
            "max_span_endpoint_prior_delta_ratio",
            _nonnegative_float(
                self.max_span_endpoint_prior_delta_ratio,
                "max_span_endpoint_prior_delta_ratio",
            ),
        )

    def span_layer_config(self) -> Lfm2ClosedLoopFoldConfig:
        """Force multi-layer correction through a bounded survivor basis."""

        existing_ratio = self.layer_config.max_endpoint_prior_delta_ratio
        ratio = self.max_span_endpoint_prior_delta_ratio
        if existing_ratio is not None:
            ratio = min(ratio, existing_ratio)
        return replace(
            self.layer_config,
            max_endpoint_prior_delta_ratio=ratio,
            # With real LFM target widths and cached row counts, these guards
            # force the bounded survivor-anchor branch. Degenerate width/row-1
            # cases remain rank one and therefore bounded by construction.
            max_primal_features=1,
            max_dual_rows=1,
            max_correction_anchors=min(
                self.layer_config.max_correction_anchors,
                self.max_span_correction_anchors,
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_config": self.layer_config.as_dict(),
            "max_span_correction_anchors": self.max_span_correction_anchors,
            "max_span_endpoint_prior_delta_ratio": (
                self.max_span_endpoint_prior_delta_ratio
            ),
            "multi_layer_solver_policy": (
                "real-layer-anchor-plus-bounded-survivor-residual-correction"
            ),
        }


@dataclass(frozen=True, slots=True)
class Lfm2AnchorScore:
    source_layer_index: int
    residual_delta_rmse: float
    residual_delta_relative_rmse: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_layer_index": self.source_layer_index,
            "residual_delta_rmse": self.residual_delta_rmse,
            "residual_delta_relative_rmse": (
                self.residual_delta_relative_rmse
            ),
        }


@dataclass(frozen=True, slots=True)
class Lfm2SpanFoldPlan:
    """One existing closed-loop layer candidate bound to a source span."""

    span: Lfm2ModelFoldSpan
    anchor_source_layer_index: int
    anchor_scores: tuple[Lfm2AnchorScore, ...]
    layer_plan: Lfm2ClosedLoopFoldPlan
    cached_span_pre_sha256: str
    cached_span_post_sha256: str
    cached_span_delta_sha256: str
    cached_span_evidence_sha256: str
    endpoint_correction_bounded: bool
    endpoint_correction_rank_bound: int
    endpoint_correction_prior_delta_ratio_bound: float | None
    authenticated_live_producer_state_keys: tuple[str, ...]

    @property
    def exact_composition_claimed(self) -> bool:
        return not self.span.is_depth_fold

    def as_dict(self) -> dict[str, Any]:
        return {
            "span": self.span.as_dict(),
            "anchor_source_layer_index": self.anchor_source_layer_index,
            "anchor_is_real_source_layer": (
                self.anchor_source_layer_index in self.span.source_layer_indices
            ),
            "anchor_selection": {
                "method": (
                    "minimum-cached-residual-delta-relative-rmse-then-source-index"
                ),
                "scores": [value.as_dict() for value in self.anchor_scores],
            },
            "depth_composition": {
                "exact_composition_claimed": self.exact_composition_claimed,
                "method": (
                    "singleton-closed-loop-layer-fold"
                    if not self.span.is_depth_fold
                    else (
                        "real-layer-anchor-plus-bounded-full-span-"
                        "residual-endpoint-correction"
                    )
                ),
                "fit_domain": (
                    "cached-span-input-to-parent-last-layer-post-residual"
                ),
                "child_side_target_design": True,
                "endpoint_correction_bounded": (
                    self.endpoint_correction_bounded
                ),
                "endpoint_correction_rank_bound": (
                    self.endpoint_correction_rank_bound
                ),
                "endpoint_correction_prior_delta_ratio_bound": (
                    self.endpoint_correction_prior_delta_ratio_bound
                ),
            },
            "cached_span_pre_sha256": self.cached_span_pre_sha256,
            "cached_span_post_sha256": self.cached_span_post_sha256,
            "cached_span_delta_sha256": self.cached_span_delta_sha256,
            "cached_span_evidence_sha256": self.cached_span_evidence_sha256,
            "layer_plan": self.layer_plan.as_dict(),
            "child_feature_producer_authentication": {
                "live_producer_shell_matches_selected_plan": True,
                "authenticated_state_keys": list(
                    self.authenticated_live_producer_state_keys
                ),
                "attention_scope": "Q/K/V producer weights",
                "mlp_scope": "coupled gate/up producer weights",
                "short_conv_scope": "complete same-shape conv state",
            },
            "cached_evidence_only": True,
            "parent_forward_count": 0,
            "zero_gradient": True,
        }


@dataclass(frozen=True, slots=True)
class Lfm2ModelFoldPlan:
    """Immutable full-child candidate installed by one model transaction."""

    spans: tuple[Lfm2ModelFoldSpan, ...]
    span_plans: tuple[Lfm2SpanFoldPlan, ...]
    config: Lfm2ModelFoldConfig
    cache_fingerprint: str
    child_feature_state_sha256: str
    evidence_sha256: str
    source_layer_contracts: tuple[Lfm2LayerContract, ...]
    target_layer_contracts: tuple[Lfm2LayerContract, ...]
    target_layer_paths: tuple[str, ...]
    copied_non_layer_state_keys: tuple[str, ...]
    candidate_state_items: tuple[tuple[str, Any], ...]
    target_state_spec: tuple[tuple[str, tuple[int, ...], str], ...]
    source_state_sha256: str
    target_state_sha256_before: str
    candidate_state_sha256: str
    source_unique_parameters: int
    target_unique_parameters: int
    parameters_removed: int
    plan_sha256: str

    @property
    def zero_gradient(self) -> bool:
        return True

    @property
    def seed_erasure_by_construction(self) -> bool:
        return True

    def candidate_state_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                name: value.detach().to(device="cpu").contiguous().clone()
                for name, value in self.candidate_state_items
            }
        )

    def _topology_summary(self) -> dict[str, Any]:
        source_attention = {
            (
                contract.operator.query_heads,
                contract.operator.key_value_heads,
                contract.operator.head_dim,
            )
            for contract in self.source_layer_contracts
            if isinstance(contract.operator, Lfm2AttentionContract)
        }
        target_attention = {
            (
                contract.operator.query_heads,
                contract.operator.key_value_heads,
                contract.operator.head_dim,
            )
            for contract in self.target_layer_contracts
            if isinstance(contract.operator, Lfm2AttentionContract)
        }
        return {
            "source_layer_count": len(self.source_layer_contracts),
            "target_layer_count": len(self.target_layer_contracts),
            "source_attention_q_kv_head_dim": [
                list(value) for value in sorted(source_attention)
            ],
            "target_attention_q_kv_head_dim": [
                list(value) for value in sorted(target_attention)
            ],
            "source_ffn_widths": sorted(
                {value.ffn_intermediate_size for value in self.source_layer_contracts}
            ),
            "target_ffn_widths": sorted(
                {value.ffn_intermediate_size for value in self.target_layer_contracts}
            ),
            "depth_width_attention_folded_simultaneously": True,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": MODEL_FOLD_SCHEMA,
            "method_id": MODEL_FOLD_METHOD,
            "spans": [value.as_dict() for value in self.spans],
            "span_plans": [value.as_dict() for value in self.span_plans],
            "config": self.config.as_dict(),
            "cache_fingerprint": self.cache_fingerprint,
            "child_feature_state_sha256": self.child_feature_state_sha256,
            "evidence_sha256": self.evidence_sha256,
            "child_feature_state_authentication": {
                "matched_live_child_during_planning": True,
                "all_target_layer_producer_shells_match_selected_plans": True,
                "two_phase_dependency": (
                    "selected producer shells must be materialized before "
                    "child-side target designs are captured"
                ),
                "design_limitation": (
                    "this fold-only primitive authenticates the cache binding "
                    "and producer shell but does not invoke the child to "
                    "independently recompute supplied feature tensors"
                ),
            },
            "topology": self._topology_summary(),
            "source_layer_span_coverage_complete": True,
            "target_layer_coverage_complete": True,
            "copied_non_layer_state_keys": list(self.copied_non_layer_state_keys),
            "copied_non_layer_state_from_parent": True,
            "candidate_state_keys": [
                name for name, _value in self.candidate_state_items
            ],
            "complete_target_state_coverage": True,
            "source_state_sha256": self.source_state_sha256,
            "target_state_sha256_before": self.target_state_sha256_before,
            "candidate_state_sha256": self.candidate_state_sha256,
            "source_unique_parameters": self.source_unique_parameters,
            "target_unique_parameters": self.target_unique_parameters,
            "parameters_removed": self.parameters_removed,
            "plan_sha256": self.plan_sha256,
            "atomic_model_level_transaction": True,
            "cached_evidence_only": True,
            "parent_forward_count": 0,
            "child_forward_count": 0,
            "seed_erasure_by_construction": True,
            "zero_gradient": True,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "training_steps": 0,
            "parameter_gradients_written": False,
        }


@dataclass(frozen=True, slots=True)
class Lfm2ModelSnapshot:
    """Exact whole-model state and runtime snapshot."""

    state_items: tuple[tuple[str, Any], ...]
    state_spec: tuple[tuple[str, tuple[int, ...], str], ...]
    state_sha256: str
    buffer_items: tuple[tuple[str, Any], ...]
    buffer_identity: tuple[tuple[str, int, tuple[int, ...], str], ...]
    parameter_ids: tuple[tuple[str, int], ...]
    gradient_signature: tuple[tuple[str, int, bool, int | None, str | None], ...]
    module_mode_signature: tuple[tuple[str, int, bool], ...]
    config_sha256: str | None

    def restore(self, target_model: Any) -> bool:
        return restore_lfm2_model_snapshot(target_model, self)


def snapshot_lfm2_model(target_model: Any) -> Lfm2ModelSnapshot:
    """Capture a complete rollback snapshot without invoking the model."""

    _decoder_layers(target_model)
    items = _state_items(target_model)
    state = _state_mapping(items)
    return Lfm2ModelSnapshot(
        state_items=items,
        state_spec=_state_spec(state),
        state_sha256=_state_sha256(state),
        buffer_items=_buffer_items(target_model),
        buffer_identity=_buffer_identity(target_model),
        parameter_ids=_parameter_ids(target_model),
        gradient_signature=_gradient_signature(target_model),
        module_mode_signature=_module_mode_signature(target_model),
        config_sha256=_model_config_sha256(target_model),
    )


def _live_named_buffers(module: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in _named_buffers(module):
        if name in result and result[name] is not value:
            raise Lfm2ModelFoldInvariantError(
                f"Duplicate buffer name {name!r} resolves to different objects."
            )
        result[name] = value
    return result


def restore_lfm2_model_snapshot(
    target_model: Any,
    snapshot: Lfm2ModelSnapshot,
) -> bool:
    """Restore state and buffers in-place and prove byte-exact rollback."""

    if not isinstance(snapshot, Lfm2ModelSnapshot):
        raise TypeError("snapshot must be Lfm2ModelSnapshot.")
    current = _live_state(target_model)
    if _state_spec(current) != snapshot.state_spec:
        raise Lfm2ModelFoldTopologyError(
            "Snapshot and current model state specifications differ."
        )
    if (
        _parameter_ids(target_model) != snapshot.parameter_ids
        or _gradient_signature(target_model) != snapshot.gradient_signature
        or _buffer_identity(target_model) != snapshot.buffer_identity
        or _module_mode_signature(target_model) != snapshot.module_mode_signature
        or _model_config_sha256(target_model) != snapshot.config_sha256
    ):
        raise Lfm2ModelFoldInvariantError(
            "Model runtime identity changed before rollback."
        )
    saved = _state_mapping(snapshot.state_items)
    buffers = _live_named_buffers(target_model)
    torch = _torch()
    with torch.no_grad():
        for name in sorted(current):
            current[name].copy_(
                saved[name].to(
                    device=current[name].device,
                    dtype=current[name].dtype,
                )
            )
        for name, value in snapshot.buffer_items:
            if name not in buffers:
                raise Lfm2ModelFoldInvariantError(
                    f"Rollback buffer {name!r} disappeared."
                )
            buffers[name].copy_(
                value.to(device=buffers[name].device, dtype=buffers[name].dtype)
            )
    return bool(
        _live_state_sha256(target_model) == snapshot.state_sha256
        and _parameter_ids(target_model) == snapshot.parameter_ids
        and _gradient_signature(target_model) == snapshot.gradient_signature
        and _buffer_identity(target_model) == snapshot.buffer_identity
        and _module_mode_signature(target_model) == snapshot.module_mode_signature
        and _model_config_sha256(target_model) == snapshot.config_sha256
    )


def _decoder_layers(model: Any) -> tuple[Any, ...]:
    core = getattr(model, "model", model)
    layers = getattr(core, "layers", None)
    try:
        result = tuple(layers)
    except TypeError as exc:
        raise Lfm2ModelFoldTopologyError(
            "Could not resolve model.model.layers for the LFM2 model."
        ) from exc
    if not result:
        raise Lfm2ModelFoldTopologyError("The LFM2 model has no decoder layers.")
    return result


def _module_path(model: Any, target: Any) -> str:
    for path, module in model.named_modules():
        if module is target and path:
            return path
    raise Lfm2ModelFoldTopologyError(
        "Could not bind an LFM2 decoder layer to its model path."
    )


def _validate_span_partition(
    spans: tuple[Lfm2ModelFoldSpan, ...],
    source_layer_count: int,
    target_layer_count: int,
) -> None:
    if len(spans) != target_layer_count:
        raise Lfm2ModelFoldTopologyError(
            "There must be exactly one source span per target layer."
        )
    target_indices = tuple(value.target_layer_index for value in spans)
    if target_indices != tuple(range(target_layer_count)):
        raise Lfm2ModelFoldTopologyError(
            "Target layers must be covered exactly once in execution order."
        )
    flattened = tuple(
        index for span in spans for index in span.source_layer_indices
    )
    if flattened != tuple(range(source_layer_count)):
        raise Lfm2ModelFoldTopologyError(
            "Source spans must exactly partition every source layer in order."
        )


@dataclass(frozen=True, slots=True)
class _ValidatedSpanRows:
    pre: Any
    post: Any
    delta: Any
    validation_pre: Any | None
    validation_post: Any | None
    validation_delta: Any | None
    layer_pre: Mapping[int, Any]
    layer_post: Mapping[int, Any]


def _validate_cached_span_rows(
    value: Lfm2CachedSpanFoldEvidence,
) -> _ValidatedSpanRows:
    pre = _float64_rows(
        value.parent_pre_span_residual_rows,
        "parent_pre_span_residual_rows",
    )
    post = _float64_rows(
        value.parent_post_span_residual_rows,
        "parent_post_span_residual_rows",
    )
    if pre.shape != post.shape:
        raise Lfm2ModelFoldEvidenceError(
            "Cached span pre/post residual shapes differ."
        )
    layer_pre: dict[int, Any] = {}
    layer_post: dict[int, Any] = {}
    previous = pre
    for position, layer in enumerate(value.layers):
        current_pre = _float64_rows(
            layer.parent_pre_residual_rows,
            f"layer {layer.source_layer_index} parent pre residual",
        )
        current_post = _float64_rows(
            layer.parent_post_residual_rows,
            f"layer {layer.source_layer_index} parent post residual",
        )
        if (
            current_pre.shape != pre.shape
            or current_post.shape != pre.shape
            or not _rows_equal(previous, current_pre)
        ):
            raise Lfm2ModelFoldEvidenceError(
                "Cached layer residuals do not form one aligned span chain."
            )
        fold_parent_post = _float64_rows(
            layer.fold_evidence.mlp.parent_branch_output_rows,
            (
                f"layer {layer.source_layer_index} fold parent "
                "post residual"
            ),
        )
        if not _rows_equal(fold_parent_post, current_post):
            raise Lfm2ModelFoldEvidenceError(
                "Layer fold evidence is not bound to its cached post residual."
            )
        child_pre = _float64_rows(
            layer.fold_evidence.mlp.child_pre_branch_residual_rows,
            f"layer {layer.source_layer_index} child pre-MLP residual",
        )
        child_design = _float64_rows(
            layer.fold_evidence.mlp.child_post_swiglu_rows,
            f"layer {layer.source_layer_index} child post-SwiGLU design",
        )
        if int(child_pre.shape[0]) != int(pre.shape[0]) or int(
            child_design.shape[0]
        ) != int(pre.shape[0]):
            raise Lfm2ModelFoldEvidenceError(
                "Child-side target designs are not aligned to cached span rows."
            )
        layer_pre[layer.source_layer_index] = current_pre
        layer_post[layer.source_layer_index] = current_post
        previous = current_post
        if position == len(value.layers) - 1 and not _rows_equal(previous, post):
            raise Lfm2ModelFoldEvidenceError(
                "The final cached layer residual is not the span post endpoint."
            )

    validation_pre = validation_post = validation_delta = None
    if value.validation_parent_pre_span_residual_rows is not None:
        validation_pre = _float64_rows(
            value.validation_parent_pre_span_residual_rows,
            "validation_parent_pre_span_residual_rows",
        )
        validation_post = _float64_rows(
            value.validation_parent_post_span_residual_rows,
            "validation_parent_post_span_residual_rows",
        )
        if validation_pre.shape != validation_post.shape:
            raise Lfm2ModelFoldEvidenceError(
                "Cached validation span pre/post residual shapes differ."
            )
        previous_validation = validation_pre
        for layer in value.layers:
            if (
                layer.validation_parent_pre_residual_rows is None
                or layer.validation_parent_post_residual_rows is None
            ):
                raise Lfm2ModelFoldEvidenceError(
                    "Span validation requires validation endpoints for every layer."
                )
            current_pre = _float64_rows(
                layer.validation_parent_pre_residual_rows,
                f"layer {layer.source_layer_index} validation pre residual",
            )
            current_post = _float64_rows(
                layer.validation_parent_post_residual_rows,
                f"layer {layer.source_layer_index} validation post residual",
            )
            if (
                current_pre.shape != validation_pre.shape
                or current_post.shape != validation_pre.shape
                or not _rows_equal(previous_validation, current_pre)
            ):
                raise Lfm2ModelFoldEvidenceError(
                    "Validation layer residuals do not form an aligned span chain."
                )
            fold_validation_post = (
                layer.fold_evidence.mlp.validation_parent_branch_output_rows
            )
            if fold_validation_post is None or not _rows_equal(
                _float64_rows(
                    fold_validation_post,
                    (
                        f"layer {layer.source_layer_index} fold validation "
                        "post residual"
                    ),
                ),
                current_post,
            ):
                raise Lfm2ModelFoldEvidenceError(
                    "Layer fold validation is not bound to its cached endpoint."
                )
            if (
                layer.fold_evidence.mlp.validation_child_post_swiglu_rows is None
                or layer.fold_evidence.mlp.validation_child_pre_branch_residual_rows
                is None
            ):
                raise Lfm2ModelFoldEvidenceError(
                    "Validation span is missing child-side validation designs."
                )
            previous_validation = current_post
        if not _rows_equal(previous_validation, validation_post):
            raise Lfm2ModelFoldEvidenceError(
                "Validation final layer residual is not the span endpoint."
            )
        validation_delta = validation_post - validation_pre
    elif any(
        layer.validation_parent_pre_residual_rows is not None
        for layer in value.layers
    ):
        raise Lfm2ModelFoldEvidenceError(
            "Layer validation endpoints require declared span validation endpoints."
        )

    return _ValidatedSpanRows(
        pre=pre,
        post=post,
        delta=post - pre,
        validation_pre=validation_pre,
        validation_post=validation_post,
        validation_delta=validation_delta,
        layer_pre=MappingProxyType(layer_pre),
        layer_post=MappingProxyType(layer_post),
    )


def _anchor_scores(
    span_evidence: Lfm2CachedSpanFoldEvidence,
    rows: _ValidatedSpanRows,
    source_contracts: tuple[Lfm2LayerContract, ...],
    target_contract: Lfm2LayerContract,
) -> tuple[Lfm2AnchorScore, ...]:
    torch = _torch()
    target_rms = float(torch.sqrt(torch.mean(rows.delta.square())).item())
    denominator = max(target_rms, torch.finfo(torch.float64).tiny)
    result = []
    for layer in span_evidence.layers:
        index = layer.source_layer_index
        if source_contracts[index].kind != target_contract.kind:
            continue
        layer_delta = rows.layer_post[index] - rows.layer_pre[index]
        rmse = float(
            torch.sqrt(torch.mean((rows.delta - layer_delta).square())).item()
        )
        result.append(
            Lfm2AnchorScore(
                source_layer_index=index,
                residual_delta_rmse=rmse,
                residual_delta_relative_rmse=rmse / denominator,
            )
        )
    if not result:
        raise Lfm2ModelFoldTopologyError(
            "A source span has no real layer compatible with its target operator."
        )
    return tuple(
        sorted(
            result,
            key=lambda value: (
                value.residual_delta_relative_rmse,
                value.source_layer_index,
            ),
        )
    )


def _span_adjusted_evidence(
    layer: Lfm2CachedLayerFoldEvidence,
    rows: _ValidatedSpanRows,
) -> Lfm2ClosedLoopFoldEvidence:
    # Express the endpoint as pre + full cached span delta.  This is the
    # parent last-layer post residual, not an average or a weight composition.
    parent_post = rows.pre + rows.delta
    validation_post = None
    if rows.validation_pre is not None:
        assert rows.validation_delta is not None
        validation_post = rows.validation_pre + rows.validation_delta
    mlp = replace(
        layer.fold_evidence.mlp,
        parent_branch_output_rows=parent_post,
        validation_parent_branch_output_rows=validation_post,
    )
    return Lfm2ClosedLoopFoldEvidence(
        mlp=mlp,
        attention=layer.fold_evidence.attention,
    )


def _span_evidence_sha256(value: Lfm2CachedSpanFoldEvidence) -> str:
    return _canonical_sha256(
        {
            "schema": MODEL_FOLD_EVIDENCE_SCHEMA,
            "span": _evidence_identity(value),
        }
    )


def _selected_producer_state(
    source_layer: Any,
    target_layer: Any,
    evidence: Lfm2ClosedLoopFoldEvidence,
    config: Lfm2ClosedLoopFoldConfig,
) -> Mapping[str, Any]:
    """Select producers without reading any child-side endpoint design."""

    torch = _torch()
    source = inspect_lfm2_layer(source_layer)
    target = inspect_lfm2_layer(target_layer)
    gate = _float64_rows(evidence.mlp.parent_gate_rows, "parent_gate_rows")
    up = _float64_rows(evidence.mlp.parent_up_rows, "parent_up_rows")
    post = _float64_rows(
        evidence.mlp.parent_post_swiglu_rows,
        "parent_post_swiglu_rows",
    )
    expected_mlp_shape = (int(gate.shape[0]), source.ffn_intermediate_size)
    if (
        tuple(gate.shape) != expected_mlp_shape
        or tuple(up.shape) != expected_mlp_shape
        or tuple(post.shape) != expected_mlp_shape
    ):
        raise Lfm2ModelFoldEvidenceError(
            "Parent MLP producer-selection rows do not match the source width."
        )
    if evidence.mlp.parent_contribution_rows is None:
        source_down = source_layer.feed_forward.w2.weight.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        mlp_contribution = post * torch.linalg.vector_norm(
            source_down,
            dim=0,
        ).unsqueeze(0)
    else:
        contribution_ndim = getattr(
            evidence.mlp.parent_contribution_rows,
            "ndim",
            None,
        )
        if contribution_ndim not in {2, 3}:
            raise Lfm2ModelFoldEvidenceError(
                "MLP parent contribution rows must have rank 2 or 3."
            )
        mlp_contribution = _float64_evidence_tensor(
            evidence.mlp.parent_contribution_rows,
            "parent_contribution_rows",
            ndim=int(contribution_ndim),
        )
    mlp_cluster = build_gated_mlp_activation_cluster_plan(
        gate,
        up,
        post,
        mlp_contribution,
        target_neurons=target.ffn_intermediate_size,
        source_neurons=source.ffn_intermediate_size,
        ridge=config.cluster_ridge,
        max_iterations=config.cluster_max_iterations,
        distance_chunk_size=config.cluster_distance_chunk_size,
    )
    result = {
        "feed_forward.w1.weight": mlp_cluster.select_gate_producer(
            source_layer.feed_forward.w1.weight,
            axis=0,
        ),
        "feed_forward.w3.weight": mlp_cluster.select_up_producer(
            source_layer.feed_forward.w3.weight,
            axis=0,
        ),
    }

    if source.kind == "full_attention":
        if (
            evidence.attention is None
            or not isinstance(source.operator, Lfm2AttentionContract)
            or not isinstance(target.operator, Lfm2AttentionContract)
        ):
            raise Lfm2ModelFoldEvidenceError(
                "Attention producer selection requires cached attention evidence."
            )
        attention = evidence.attention
        query = _float64_evidence_tensor(
            attention.parent_query_rows,
            "parent_query_rows",
            ndim=3,
        )
        key = _float64_evidence_tensor(
            attention.parent_key_rows,
            "parent_key_rows",
            ndim=3,
        )
        value = _float64_evidence_tensor(
            attention.parent_value_rows,
            "parent_value_rows",
            ndim=3,
        )
        context = _float64_evidence_tensor(
            attention.parent_context_rows,
            "parent_context_rows",
            ndim=3,
        )
        if attention.parent_contribution_rows is None:
            output_blocks = (
                source_layer.self_attn.out_proj.weight.detach()
                .to(device="cpu", dtype=torch.float64)
                .reshape(
                    source.hidden_size,
                    source.operator.query_heads,
                    source.operator.head_dim,
                )
            )
            attention_contribution = torch.einsum(
                "rqd,hqd->rqh",
                context,
                output_blocks,
            )
        else:
            attention_contribution = _float64_evidence_tensor(
                attention.parent_contribution_rows,
                "attention parent_contribution_rows",
                ndim=3,
            )
        attention_cluster = build_gqa_activation_cluster_plan(
            query.movedim(1, 0).contiguous(),
            key.movedim(1, 0).contiguous(),
            value.movedim(1, 0).contiguous(),
            context.movedim(1, 0).contiguous(),
            attention_contribution.movedim(1, 0).contiguous(),
            source_query_heads=source.operator.query_heads,
            source_key_value_heads=source.operator.key_value_heads,
            target_query_heads=target.operator.query_heads,
            target_key_value_heads=target.operator.key_value_heads,
            source_head_dim=source.operator.head_dim,
            target_head_dim=target.operator.head_dim,
            ridge=config.cluster_ridge,
            max_iterations=config.cluster_max_iterations,
            distance_chunk_size=config.cluster_distance_chunk_size,
        )
        result.update(
            {
                "self_attn.q_proj.weight": (
                    attention_cluster.select_query_producer(
                        source_layer.self_attn.q_proj.weight,
                        axis=0,
                    )
                ),
                "self_attn.k_proj.weight": (
                    attention_cluster.select_key_producer(
                        source_layer.self_attn.k_proj.weight,
                        axis=0,
                    )
                ),
                "self_attn.v_proj.weight": (
                    attention_cluster.select_value_producer(
                        source_layer.self_attn.v_proj.weight,
                        axis=0,
                    )
                ),
            }
        )
    else:
        source_state = source_layer.state_dict()
        target_state = target_layer.state_dict()
        conv_keys = tuple(
            sorted(name for name in target_state if name.startswith("conv."))
        )
        if not conv_keys or any(name not in source_state for name in conv_keys):
            raise Lfm2ModelFoldTopologyError(
                "Short-convolution producer state is incomplete."
            )
        result.update({name: source_state[name] for name in conv_keys})
    return MappingProxyType(
        {
            name: _clone_tensor(value, f"selected producer[{name!r}]")
            for name, value in sorted(result.items())
        }
    )


def _authenticate_live_producer_shell(
    target_layer: Any,
    expected: Mapping[str, Any],
) -> tuple[str, ...]:
    """Reject stale child designs before any endpoint fit is accepted."""

    live = target_layer.state_dict()
    required = tuple(sorted(expected))
    missing = sorted(set(required).difference(live))
    if missing:
        raise Lfm2ModelFoldTopologyError(
            "The target layer is missing required producer state: "
            + ", ".join(missing)
        )
    mismatched = []
    torch = _torch()
    for name in required:
        observed = live[name].detach().to(device="cpu").contiguous()
        selected = expected[name].detach().to(
            device="cpu",
            dtype=observed.dtype,
        ).contiguous()
        if observed.shape != selected.shape or not bool(
            torch.equal(observed, selected)
        ):
            mismatched.append(name)
    if mismatched:
        raise Lfm2ModelFoldEvidenceError(
            "The authenticated child feature state is not the selected producer "
            "shell; cached child designs from a zero/random blank cannot be used. "
            "Mismatched producer state: "
            + ", ".join(mismatched)
        )
    return required


def _assert_plan_uses_authenticated_producers(
    plan: Lfm2ClosedLoopFoldPlan,
    expected: Mapping[str, Any],
) -> None:
    candidate = plan.candidate_state_dict()
    torch = _torch()
    for name, selected in expected.items():
        if name not in candidate:
            raise Lfm2ModelFoldInvariantError(
                f"Closed-loop plan lost authenticated producer {name!r}."
            )
        planned = candidate[name].detach().to(device="cpu").contiguous()
        normalized = selected.detach().to(
            device="cpu",
            dtype=planned.dtype,
        ).contiguous()
        if planned.shape != normalized.shape or not bool(
            torch.equal(planned, normalized)
        ):
            raise Lfm2ModelFoldInvariantError(
                "Closed-loop endpoint fitting changed an authenticated producer."
            )


def _validate_global_residual_chain(
    rows: tuple[_ValidatedSpanRows, ...],
) -> None:
    for previous, current in zip(rows, rows[1:]):
        if not _rows_equal(previous.post, current.pre):
            raise Lfm2ModelFoldEvidenceError(
                "Cached span endpoints do not form one full-model residual chain."
            )
        previous_has_validation = previous.validation_post is not None
        current_has_validation = current.validation_pre is not None
        if previous_has_validation != current_has_validation:
            raise Lfm2ModelFoldEvidenceError(
                "Validation residual coverage changes between adjacent spans."
            )
        if previous_has_validation and not _rows_equal(
            previous.validation_post,
            current.validation_pre,
        ):
            raise Lfm2ModelFoldEvidenceError(
                "Validation span endpoints do not form a full-model chain."
            )


def _candidate_for_model(
    parent_model: Any,
    child_model: Any,
    target_layer_paths: tuple[str, ...],
    span_plans: tuple[Lfm2SpanFoldPlan, ...],
) -> tuple[tuple[tuple[str, Any], ...], tuple[str, ...]]:
    parent = _live_state(parent_model)
    target = _live_state(child_model)
    layer_candidates = tuple(
        value.layer_plan.candidate_state_dict() for value in span_plans
    )
    candidate: dict[str, Any] = {}
    copied_non_layer = []
    for name in sorted(target):
        owner = None
        local_name = None
        for index, prefix in enumerate(target_layer_paths):
            marker = prefix + "."
            if name.startswith(marker):
                owner = index
                local_name = name[len(marker) :]
                break
        if owner is not None:
            assert local_name is not None
            local = layer_candidates[owner]
            if local_name not in local:
                raise Lfm2ModelFoldTopologyError(
                    f"Layer plan does not own target state {name!r}."
                )
            value = local[local_name]
        else:
            if name not in parent:
                raise Lfm2ModelFoldTopologyError(
                    f"Parent state has no non-layer target tensor {name!r}."
                )
            value = parent[name]
            copied_non_layer.append(name)
        target_value = target[name]
        if (
            tuple(value.shape) != tuple(target_value.shape)
            or value.dtype != target_value.dtype
        ):
            raise Lfm2ModelFoldTopologyError(
                f"Candidate tensor {name!r} does not match the target state spec."
            )
        candidate[name] = _clone_tensor(value, f"candidate[{name!r}]")
    if set(candidate) != set(target):
        raise Lfm2ModelFoldInvariantError(
            "The model candidate does not overwrite every target state tensor."
        )
    return (
        tuple((name, candidate[name]) for name in sorted(candidate)),
        tuple(copied_non_layer),
    )


def _runtime_signature(module: Any) -> tuple[Any, ...]:
    return (
        _parameter_ids(module),
        _gradient_signature(module),
        _buffer_identity(module),
        _module_mode_signature(module),
        _model_config_sha256(module),
    )


def build_lfm2_model_fold_plan(
    parent_model: Any,
    child_model: Any,
    spans: Sequence[Lfm2ModelFoldSpan],
    evidence: Lfm2ModelFoldEvidence,
    *,
    config: Lfm2ModelFoldConfig | None = None,
) -> Lfm2ModelFoldPlan:
    """Build a complete child candidate without invoking or mutating a model."""

    selected = Lfm2ModelFoldConfig() if config is None else config
    if not isinstance(selected, Lfm2ModelFoldConfig):
        raise TypeError("config must be Lfm2ModelFoldConfig or None.")
    if not isinstance(evidence, Lfm2ModelFoldEvidence):
        raise TypeError("evidence must be Lfm2ModelFoldEvidence.")
    try:
        span_values = tuple(spans)
    except TypeError as exc:
        raise Lfm2ModelFoldTopologyError(
            "spans must be an LFM2 model-fold span sequence."
        ) from exc
    if not span_values or any(
        not isinstance(value, Lfm2ModelFoldSpan) for value in span_values
    ):
        raise Lfm2ModelFoldTopologyError(
            "spans must contain Lfm2ModelFoldSpan values."
        )

    source_layers = _decoder_layers(parent_model)
    target_layers = _decoder_layers(child_model)
    _validate_span_partition(
        span_values,
        len(source_layers),
        len(target_layers),
    )
    evidence_by_target = {
        value.span.target_layer_index: value for value in evidence.spans
    }
    if tuple(sorted(evidence_by_target)) != tuple(range(len(target_layers))):
        raise Lfm2ModelFoldEvidenceError(
            "Model evidence does not exactly cover every target span."
        )
    for span in span_values:
        if evidence_by_target[span.target_layer_index].span != span:
            raise Lfm2ModelFoldEvidenceError(
                "Model evidence span definitions differ from the requested plan."
            )

    source_contracts = tuple(inspect_lfm2_layer(layer) for layer in source_layers)
    target_contracts = tuple(inspect_lfm2_layer(layer) for layer in target_layers)
    target_layer_paths = tuple(
        _module_path(child_model, layer) for layer in target_layers
    )
    source_state_before = _live_state_sha256(parent_model)
    target_state_before = _live_state_sha256(child_model)
    if evidence.child_feature_state_sha256 != target_state_before:
        raise Lfm2ModelFoldEvidenceError(
            "child_feature_state_sha256 does not match the live child state used "
            "for planning; cached child-side designs are stale or unbound."
        )
    source_runtime_before = _runtime_signature(parent_model)
    target_runtime_before = _runtime_signature(child_model)
    evidence_before = fingerprint_lfm2_model_fold_evidence(evidence)

    validated_rows = tuple(
        _validate_cached_span_rows(evidence_by_target[index])
        for index in range(len(target_layers))
    )
    _validate_global_residual_chain(validated_rows)

    span_plans = []
    torch = _torch()
    with torch.no_grad():
        for target_index, span in enumerate(span_values):
            cached = evidence_by_target[target_index]
            rows = validated_rows[target_index]
            scores = _anchor_scores(
                cached,
                rows,
                source_contracts,
                target_contracts[target_index],
            )
            anchor_index = scores[0].source_layer_index
            anchor_evidence = next(
                value
                for value in cached.layers
                if value.source_layer_index == anchor_index
            )
            fold_evidence = anchor_evidence.fold_evidence
            fold_config = selected.layer_config
            if span.is_depth_fold:
                fold_evidence = _span_adjusted_evidence(anchor_evidence, rows)
                fold_config = selected.span_layer_config()
            expected_producers = _selected_producer_state(
                source_layers[anchor_index],
                target_layers[target_index],
                fold_evidence,
                fold_config,
            )
            authenticated_producer_keys = _authenticate_live_producer_shell(
                target_layers[target_index],
                expected_producers,
            )
            layer_plan = build_lfm2_closed_loop_fold_plan(
                source_layers[anchor_index],
                target_layers[target_index],
                fold_evidence,
                config=fold_config,
            )
            _assert_plan_uses_authenticated_producers(
                layer_plan,
                expected_producers,
            )
            endpoint = layer_plan.mlp.endpoint_fit
            correction_bounded = not span.is_depth_fold
            ratio_bound = fold_config.max_endpoint_prior_delta_ratio
            if span.is_depth_fold:
                correction_bounded = (
                    ratio_bound is not None
                    and endpoint.correction_rank_bound
                    <= selected.max_span_correction_anchors
                    and endpoint.fit.prior_delta_norm
                    <= ratio_bound
                    * max(
                        endpoint.fit.prior_norm,
                        torch.finfo(torch.float64).tiny,
                    )
                    + 1e-12
                )
                if not correction_bounded:
                    raise Lfm2ModelFoldInvariantError(
                        "Multi-layer residual endpoint correction exceeded its "
                        "declared rank or norm bound."
                    )
            span_plans.append(
                Lfm2SpanFoldPlan(
                    span=span,
                    anchor_source_layer_index=anchor_index,
                    anchor_scores=scores,
                    layer_plan=layer_plan,
                    cached_span_pre_sha256=_tensor_sha256(rows.pre),
                    cached_span_post_sha256=_tensor_sha256(rows.post),
                    cached_span_delta_sha256=_tensor_sha256(rows.delta),
                    cached_span_evidence_sha256=_span_evidence_sha256(cached),
                    endpoint_correction_bounded=correction_bounded,
                    endpoint_correction_rank_bound=(
                        endpoint.correction_rank_bound
                    ),
                    endpoint_correction_prior_delta_ratio_bound=ratio_bound,
                    authenticated_live_producer_state_keys=(
                        authenticated_producer_keys
                    ),
                )
            )
        span_plan_values = tuple(span_plans)
        candidate_items, copied_non_layer = _candidate_for_model(
            parent_model,
            child_model,
            target_layer_paths,
            span_plan_values,
        )

    if (
        _live_state_sha256(parent_model) != source_state_before
        or _live_state_sha256(child_model) != target_state_before
        or _runtime_signature(parent_model) != source_runtime_before
        or _runtime_signature(child_model) != target_runtime_before
    ):
        raise Lfm2ModelFoldInvariantError(
            "Building the model fold mutated a model or runtime state."
        )
    evidence_after = fingerprint_lfm2_model_fold_evidence(evidence)
    if evidence_after != evidence_before:
        raise Lfm2ModelFoldInvariantError(
            "Cached model-fold evidence changed during planning."
        )

    candidate = _state_mapping(candidate_items)
    target_spec = _state_spec(_live_state(child_model))
    candidate_sha = _state_sha256(candidate)
    source_parameters = _unique_parameter_count(parent_model)
    target_parameters = _unique_parameter_count(child_model)
    removed = source_parameters - target_parameters
    if removed <= 0:
        raise Lfm2ModelFoldTopologyError(
            "The target model does not remove learned parameters."
        )
    provenance = {
        "schema": MODEL_FOLD_SCHEMA,
        "method_id": MODEL_FOLD_METHOD,
        "spans": [value.as_dict() for value in span_values],
        "span_plans": [value.as_dict() for value in span_plan_values],
        "config": selected.as_dict(),
        "cache_fingerprint": evidence.cache_fingerprint,
        "child_feature_state_sha256": (
            evidence.child_feature_state_sha256
        ),
        "evidence_sha256": evidence_before,
        "source_contracts": [value.as_dict() for value in source_contracts],
        "target_contracts": [value.as_dict() for value in target_contracts],
        "target_layer_paths": list(target_layer_paths),
        "copied_non_layer_state_keys": list(copied_non_layer),
        "source_state_sha256": source_state_before,
        "target_state_sha256_before": target_state_before,
        "candidate_state_sha256": candidate_sha,
        "source_unique_parameters": source_parameters,
        "target_unique_parameters": target_parameters,
        "parameters_removed": removed,
        "complete_source_span_coverage": True,
        "complete_target_state_coverage": True,
        "atomic_model_level_transaction": True,
        "cached_evidence_only": True,
        "zero_gradient": True,
    }
    return Lfm2ModelFoldPlan(
        spans=span_values,
        span_plans=span_plan_values,
        config=selected,
        cache_fingerprint=evidence.cache_fingerprint,
        child_feature_state_sha256=evidence.child_feature_state_sha256,
        evidence_sha256=evidence_before,
        source_layer_contracts=source_contracts,
        target_layer_contracts=target_contracts,
        target_layer_paths=target_layer_paths,
        copied_non_layer_state_keys=copied_non_layer,
        candidate_state_items=candidate_items,
        target_state_spec=target_spec,
        source_state_sha256=source_state_before,
        target_state_sha256_before=target_state_before,
        candidate_state_sha256=candidate_sha,
        source_unique_parameters=source_parameters,
        target_unique_parameters=target_parameters,
        parameters_removed=removed,
        plan_sha256=_canonical_sha256(provenance),
    )


@dataclass(frozen=True, slots=True)
class Lfm2ModelFoldTransactionReport:
    accepted: bool
    rollback_exact: bool | None
    candidate_state_sha256: str
    final_state_sha256: str
    parameter_identities_preserved: bool
    gradient_state_preserved: bool
    buffer_identities_preserved: bool
    module_modes_preserved: bool
    config_preserved: bool
    seed_erasure_by_construction: bool
    source_unique_parameters: int
    target_unique_parameters: int
    parameters_removed: int
    cache_fingerprint: str
    child_feature_state_sha256: str
    evidence_sha256: str
    plan_sha256: str

    @property
    def zero_gradient(self) -> bool:
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": MODEL_FOLD_TRANSACTION_SCHEMA,
            "method_id": MODEL_FOLD_METHOD,
            "accepted": self.accepted,
            "rollback_exact": self.rollback_exact,
            "candidate_state_sha256": self.candidate_state_sha256,
            "final_state_sha256": self.final_state_sha256,
            "parameter_identities_preserved": (
                self.parameter_identities_preserved
            ),
            "gradient_state_preserved": self.gradient_state_preserved,
            "buffer_identities_preserved": self.buffer_identities_preserved,
            "module_modes_preserved": self.module_modes_preserved,
            "config_preserved": self.config_preserved,
            "seed_erasure_by_construction": (
                self.seed_erasure_by_construction
            ),
            "source_unique_parameters": self.source_unique_parameters,
            "target_unique_parameters": self.target_unique_parameters,
            "parameters_removed": self.parameters_removed,
            "cache_fingerprint": self.cache_fingerprint,
            "child_feature_state_sha256": self.child_feature_state_sha256,
            "evidence_sha256": self.evidence_sha256,
            "plan_sha256": self.plan_sha256,
            "atomic_model_level_transaction": True,
            "cached_evidence_only": True,
            "parent_forward_count": 0,
            "child_forward_count": 0,
            "zero_gradient": True,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "training_steps": 0,
            "parameter_gradients_written": False,
        }


class Lfm2ModelFoldTransaction:
    """One installed full-model candidate awaiting commit or rollback."""

    method_id = MODEL_FOLD_METHOD

    def __init__(
        self,
        target_model: Any,
        plan: Lfm2ModelFoldPlan,
        snapshot: Lfm2ModelSnapshot,
    ) -> None:
        self._target_model = target_model
        self.plan = plan
        self.snapshot = snapshot
        self._active = True
        self._report: Lfm2ModelFoldTransactionReport | None = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def report(self) -> Lfm2ModelFoldTransactionReport | None:
        return self._report

    def _runtime_flags(self) -> tuple[bool, bool, bool, bool, bool]:
        return (
            _parameter_ids(self._target_model) == self.snapshot.parameter_ids,
            _gradient_signature(self._target_model)
            == self.snapshot.gradient_signature,
            _buffer_identity(self._target_model) == self.snapshot.buffer_identity,
            _module_mode_signature(self._target_model)
            == self.snapshot.module_mode_signature,
            _model_config_sha256(self._target_model) == self.snapshot.config_sha256,
        )

    def _make_report(
        self,
        *,
        accepted: bool,
        rollback_exact: bool | None,
        final_state_sha256: str,
    ) -> Lfm2ModelFoldTransactionReport:
        parameter_ids, gradients, buffers, modes, config = self._runtime_flags()
        return Lfm2ModelFoldTransactionReport(
            accepted=accepted,
            rollback_exact=rollback_exact,
            candidate_state_sha256=self.plan.candidate_state_sha256,
            final_state_sha256=final_state_sha256,
            parameter_identities_preserved=parameter_ids,
            gradient_state_preserved=gradients,
            buffer_identities_preserved=buffers,
            module_modes_preserved=modes,
            config_preserved=config,
            seed_erasure_by_construction=True,
            source_unique_parameters=self.plan.source_unique_parameters,
            target_unique_parameters=self.plan.target_unique_parameters,
            parameters_removed=self.plan.parameters_removed,
            cache_fingerprint=self.plan.cache_fingerprint,
            child_feature_state_sha256=(
                self.plan.child_feature_state_sha256
            ),
            evidence_sha256=self.plan.evidence_sha256,
            plan_sha256=self.plan.plan_sha256,
        )

    def commit(self) -> Lfm2ModelFoldTransactionReport:
        if not self._active:
            raise Lfm2ModelFoldInvariantError(
                "The LFM2 model-fold transaction is no longer active."
            )
        final_sha = _live_state_sha256(self._target_model)
        flags = self._runtime_flags()
        if final_sha != self.plan.candidate_state_sha256:
            raise Lfm2ModelFoldInvariantError(
                "The installed full-model candidate changed before commit."
            )
        if not all(flags):
            raise Lfm2ModelFoldInvariantError(
                "The full-model fold changed a forbidden runtime invariant."
            )
        self._active = False
        self._report = self._make_report(
            accepted=True,
            rollback_exact=None,
            final_state_sha256=final_sha,
        )
        return self._report

    def rollback(self) -> Lfm2ModelFoldTransactionReport:
        if not self._active:
            raise Lfm2ModelFoldInvariantError(
                "The LFM2 model-fold transaction is no longer active."
            )
        exact = self.snapshot.restore(self._target_model)
        final_sha = _live_state_sha256(self._target_model)
        exact = bool(
            exact
            and final_sha == self.snapshot.state_sha256
            and all(self._runtime_flags())
        )
        if not exact:
            raise Lfm2ModelFoldInvariantError(
                "The full-model candidate did not roll back byte-exactly."
            )
        self._active = False
        self._report = self._make_report(
            accepted=False,
            rollback_exact=True,
            final_state_sha256=final_sha,
        )
        return self._report

    def __enter__(self) -> "Lfm2ModelFoldTransaction":
        if not self._active:
            raise Lfm2ModelFoldInvariantError(
                "Cannot enter an inactive LFM2 model-fold transaction."
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self._active:
            self.rollback()
        return False


def install_lfm2_model_fold_plan(
    target_model: Any,
    plan: Lfm2ModelFoldPlan,
) -> Lfm2ModelFoldTransaction:
    """Install every planned tensor under one exact whole-model snapshot."""

    if not isinstance(plan, Lfm2ModelFoldPlan):
        raise TypeError("plan must be Lfm2ModelFoldPlan.")
    target_layers = _decoder_layers(target_model)
    contracts = tuple(inspect_lfm2_layer(layer) for layer in target_layers)
    paths = tuple(_module_path(target_model, layer) for layer in target_layers)
    if (
        tuple(value.as_dict() for value in contracts)
        != tuple(value.as_dict() for value in plan.target_layer_contracts)
        or paths != plan.target_layer_paths
    ):
        raise Lfm2ModelFoldTopologyError(
            "The target model topology does not match the fold plan."
        )
    current = _live_state(target_model)
    if _state_spec(current) != plan.target_state_spec:
        raise Lfm2ModelFoldTopologyError(
            "The target model state spec does not match the fold plan."
        )
    candidate = plan.candidate_state_dict()
    if _state_sha256(candidate) != plan.candidate_state_sha256:
        raise Lfm2ModelFoldInvariantError(
            "The immutable full-model candidate changed after planning."
        )
    snapshot = snapshot_lfm2_model(target_model)
    torch = _torch()
    try:
        with torch.no_grad():
            for name in sorted(current):
                current[name].copy_(
                    candidate[name].to(
                        device=current[name].device,
                        dtype=current[name].dtype,
                    )
                )
        if _live_state_sha256(target_model) != plan.candidate_state_sha256:
            raise Lfm2ModelFoldInvariantError(
                "Installed model state differs from the planned candidate."
            )
        if (
            _parameter_ids(target_model) != snapshot.parameter_ids
            or _gradient_signature(target_model) != snapshot.gradient_signature
            or _buffer_identity(target_model) != snapshot.buffer_identity
            or _module_mode_signature(target_model)
            != snapshot.module_mode_signature
            or _model_config_sha256(target_model) != snapshot.config_sha256
        ):
            raise Lfm2ModelFoldInvariantError(
                "Installing the model candidate changed a runtime invariant."
            )
    except BaseException:
        snapshot.restore(target_model)
        raise
    return Lfm2ModelFoldTransaction(target_model, plan, snapshot)


def begin_lfm2_model_fold_transaction(
    parent_model: Any,
    child_model: Any,
    spans: Sequence[Lfm2ModelFoldSpan],
    evidence: Lfm2ModelFoldEvidence,
    *,
    config: Lfm2ModelFoldConfig | None = None,
) -> Lfm2ModelFoldTransaction:
    """Build and atomically install a full model candidate."""

    plan = build_lfm2_model_fold_plan(
        parent_model,
        child_model,
        spans,
        evidence,
        config=config,
    )
    return install_lfm2_model_fold_plan(child_model, plan)


__all__ = [
    "MODEL_FOLD_EVIDENCE_SCHEMA",
    "MODEL_FOLD_METHOD",
    "MODEL_FOLD_SCHEMA",
    "MODEL_FOLD_TRANSACTION_SCHEMA",
    "Lfm2AnchorScore",
    "Lfm2CachedLayerFoldEvidence",
    "Lfm2CachedSpanFoldEvidence",
    "Lfm2ModelFoldConfig",
    "Lfm2ModelFoldError",
    "Lfm2ModelFoldEvidence",
    "Lfm2ModelFoldEvidenceError",
    "Lfm2ModelFoldInvariantError",
    "Lfm2ModelFoldPlan",
    "Lfm2ModelFoldSpan",
    "Lfm2ModelFoldTopologyError",
    "Lfm2ModelFoldTransaction",
    "Lfm2ModelFoldTransactionReport",
    "Lfm2ModelSnapshot",
    "Lfm2SpanFoldPlan",
    "assemble_lfm2_model_fold_evidence",
    "assemble_lfm2_model_fold_evidence_from_trace_rows",
    "begin_lfm2_model_fold_transaction",
    "build_leading_pair_lfm2_span_plan",
    "build_lfm2_model_fold_plan",
    "fingerprint_lfm2_model_fold_evidence",
    "fingerprint_lfm2_model_feature_state",
    "install_lfm2_model_fold_plan",
    "restore_lfm2_model_snapshot",
    "snapshot_lfm2_model",
]
