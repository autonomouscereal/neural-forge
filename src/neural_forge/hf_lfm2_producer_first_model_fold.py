"""Producer-first, full-depth LFM2 residual-attention model folding.

This is the production form of the sequence proven by
``run_lfm2_residual_attention_fold.py``.  It deliberately does not import the
experiment runner.  One transaction:

1. derives real medoid producers from immutable parent trace rows;
2. overwrites every tensor in the compiler-zero child from parent/producers;
3. captures the installed child (never the blank) on cached calibration inputs;
4. refines every layer in order and recaptures after every accepted update; and
5. exposes one whole-model commit/rollback boundary to the dynamic compiler.

No parent forward, optimizer, backward call, or gradient write is performed.
Depth folding is intentionally unsupported here; parent and child must have the
same ordered LFM2 layer topology.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .hf_activation_cluster import (
    GatedMLPActivationClusterPlan,
    GQAActivationClusterPlan,
    build_gated_mlp_activation_cluster_plan,
    build_gqa_activation_cluster_plan,
)
from .hf_lfm2_canonical import (
    Lfm2AttentionContract,
    Lfm2ShortConvContract,
    inspect_lfm2_layer,
)
from .hf_lfm2_closed_loop_fold import (
    Lfm2AttentionFoldEvidence,
    Lfm2ClosedLoopFoldConfig,
    Lfm2ClosedLoopFoldEvidence,
    Lfm2ClosedLoopFoldPlan,
    Lfm2MLPFoldEvidence,
    build_lfm2_closed_loop_fold_plan,
    install_lfm2_closed_loop_fold_plan,
)
from .hf_lfm2_circuit_trace_cache import fingerprint_lfm2_parent


PRODUCER_FIRST_MODEL_METHOD = "lfm2_producer_first_residual_attention_model"
PRODUCER_FIRST_SCHEMA = "neural-forge-lfm2-producer-first-model-fold-v1"
PRODUCER_FIRST_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-producer-first-model-evidence-v2"
)


class Lfm2ProducerFirstFoldError(RuntimeError):
    """The producer-first model fold could not prove its contract."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise Lfm2ProducerFirstFoldError(
            "PyTorch is required for LFM2 producer-first folding."
        ) from exc
    return torch


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2ProducerFirstFoldError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2ProducerFirstFoldError(f"{name} must be a positive integer.")
    return value


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Lfm2ProducerFirstFoldError(
            "Producer-first provenance is not canonical JSON."
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise Lfm2ProducerFirstFoldError(
            "Producer-first state requires materialized dense tensors."
        )
    tensor = value.detach().to(device="cpu").contiguous()
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
    digest.update(tensor.view(torch.uint8).reshape(-1).numpy().tobytes())
    return digest.hexdigest()


def _state_sha256(model: Any) -> str:
    state = model.state_dict()
    if not isinstance(state, Mapping) or not state:
        raise Lfm2ProducerFirstFoldError("LFM2 model state must not be empty.")
    digest = hashlib.sha256()
    digest.update(b"neural-forge-lfm2-producer-first-state-v1\0")
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_tensor_sha256(state[name])))
    return digest.hexdigest()


def _clone_state(model: Any) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            name: value.detach().to(device="cpu").contiguous().clone()
            for name, value in model.state_dict().items()
        }
    )


def _restore_state(model: Any, snapshot: Mapping[str, Any]) -> None:
    live = model.state_dict()
    if set(live) != set(snapshot):
        raise Lfm2ProducerFirstFoldError(
            "Whole-model rollback state keys changed."
        )
    torch = _torch()
    with torch.no_grad():
        for name, target in live.items():
            source = snapshot[name]
            if tuple(target.shape) != tuple(source.shape) or target.dtype != source.dtype:
                raise Lfm2ProducerFirstFoldError(
                    f"Whole-model rollback contract changed for {name!r}."
                )
            target.copy_(source.to(device=target.device))


def _layers(model: Any) -> tuple[Any, ...]:
    body = getattr(model, "model", None)
    layers = getattr(body, "layers", None)
    try:
        result = tuple(layers)
    except TypeError as exc:
        raise Lfm2ProducerFirstFoldError(
            "Expected model.model.layers on an LFM2 causal model."
        ) from exc
    if not result:
        raise Lfm2ProducerFirstFoldError("The LFM2 model has no layers.")
    return result


def _layer_prefixes(model: Any) -> Mapping[int, str]:
    layers = _layers(model)
    by_id = {id(layer): index for index, layer in enumerate(layers)}
    result = {
        by_id[id(module)]: name
        for name, module in model.named_modules()
        if id(module) in by_id
    }
    if set(result) != set(range(len(layers))) or any(not value for value in result.values()):
        raise Lfm2ProducerFirstFoldError(
            "Could not bind every LFM2 layer to a state-dict prefix."
        )
    return MappingProxyType(result)


def _index_rows(value: Any, indices: Sequence[int]) -> Any:
    torch = _torch()
    index = torch.tensor(tuple(indices), dtype=torch.long, device=value.device)
    return value.index_select(0, index).detach().to(device="cpu").contiguous()


def _optional_rows(value: Any, indices: Sequence[int]) -> Any | None:
    return None if not indices else _index_rows(value, indices)


def _attention_contributions(
    context_rows: Any,
    output_weight: Any,
    query_heads: int,
    head_dim: int,
) -> Any:
    torch = _torch()
    hidden_size = int(output_weight.shape[0])
    blocks = (
        output_weight.detach()
        .to(device="cpu", dtype=torch.float64)
        .reshape(hidden_size, query_heads, head_dim)
    )
    context = context_rows.detach().to(device="cpu", dtype=torch.float64)
    return torch.einsum("rqd,hqd->rqh", context, blocks).contiguous()


def _rows_metadata(rows: Any) -> tuple[tuple[str, ...], tuple[str, ...], str, str]:
    paths = tuple(str(value) for value in getattr(rows, "paths", ()))
    splits = tuple(str(value) for value in getattr(rows, "path_splits", ()))
    corpus = _digest(getattr(rows, "corpus_fingerprint", None), "corpus_fingerprint")
    trace = _digest(getattr(rows, "trace_rows_sha256", None), "trace_rows_sha256")
    if not paths or len(paths) != len(splits):
        raise Lfm2ProducerFirstFoldError(
            "Trace rows need aligned nonempty paths and path splits."
        )
    if not callable(getattr(rows, "field", None)):
        raise Lfm2ProducerFirstFoldError("Trace rows must expose field().")
    return paths, splits, corpus, trace


def _rows_identity_sha256(rows: Any) -> str:
    """Hash the exact materialized trace rows, not only their declared digest."""

    paths, splits, corpus, trace = _rows_metadata(rows)
    cache_key = _digest(
        getattr(rows, "cache_key", None),
        "parent_rows.cache_key",
    )
    layer_kinds = tuple(str(value) for value in getattr(rows, "layer_kinds", ()))
    layer_fields = getattr(rows, "layer_fields", None)
    if not layer_kinds or not isinstance(layer_fields, Mapping):
        raise Lfm2ProducerFirstFoldError(
            "Trace rows must expose nonempty layer_kinds and layer_fields."
        )
    if set(layer_fields) != set(range(len(layer_kinds))):
        raise Lfm2ProducerFirstFoldError(
            "Trace row fields must cover every ordered layer exactly once."
        )
    tensors: list[dict[str, Any]] = []
    for layer_index, kind in enumerate(layer_kinds):
        values = layer_fields[layer_index]
        if not isinstance(values, Mapping) or not values:
            raise Lfm2ProducerFirstFoldError(
                f"Trace layer {layer_index} has no materialized fields."
            )
        for name in sorted(values):
            if not isinstance(name, str) or not name:
                raise Lfm2ProducerFirstFoldError(
                    "Trace field names must be nonempty strings."
                )
            value = rows.field(layer_index, name)
            tensors.append(
                {
                    "layer_index": layer_index,
                    "layer_kind": kind,
                    "field": name,
                    "tensor_sha256": _tensor_sha256(value),
                }
            )
    return _canonical_sha256(
        {
            "schema": "neural-forge-lfm2-producer-first-trace-rows-v1",
            "cache_key": cache_key,
            "corpus_fingerprint": corpus,
            "trace_rows_sha256": trace,
            "ordered_paths_sha256": _canonical_sha256(list(paths)),
            "ordered_splits_sha256": _canonical_sha256(list(splits)),
            "layer_kinds": list(layer_kinds),
            "tensors": tensors,
        }
    )


def _live_parent_identity_sha256(parent: Any) -> str:
    try:
        return _digest(
            fingerprint_lfm2_parent(parent).identity_sha256,
            "live LFM2 parent identity",
        )
    except Lfm2ProducerFirstFoldError:
        raise
    except Exception as exc:
        raise Lfm2ProducerFirstFoldError(
            "Could not authenticate the exact loaded LFM2 parent."
        ) from exc


@dataclass(frozen=True, slots=True)
class Lfm2ProducerFirstCapture:
    """One immutable observation of the installed live child."""

    rows: Any
    cache_key: str
    child_forward_count: int
    cache_reused: bool = False

    def __post_init__(self) -> None:
        _rows_metadata(self.rows)
        object.__setattr__(self, "cache_key", _digest(self.cache_key, "cache_key"))
        if (
            isinstance(self.child_forward_count, bool)
            or not isinstance(self.child_forward_count, int)
            or self.child_forward_count < 0
        ):
            raise Lfm2ProducerFirstFoldError(
                "child_forward_count must be a non-negative integer."
            )
        if not isinstance(self.cache_reused, bool):
            raise Lfm2ProducerFirstFoldError("cache_reused must be boolean.")


@dataclass(frozen=True, slots=True)
class Lfm2ProducerFirstEvidence:
    """Sealed parent rows and a child-only capture boundary.

    Constructing this payload performs no child forward.  ``capture_child`` is
    invoked only after the complete provisional producer state is installed.
    """

    parent_rows: Any
    selection_indices: tuple[int, ...]
    fit_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    capture_child: Callable[[Any], Lfm2ProducerFirstCapture]
    parent_cache_fingerprint: str
    parent_identity_sha256: str
    capture_builder_id: str
    parent_rows_identity_sha256: str = field(init=False)
    evidence_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        paths, _splits, _corpus, _trace = _rows_metadata(self.parent_rows)
        for name in (
            "selection_indices",
            "fit_indices",
            "validation_indices",
        ):
            values = tuple(getattr(self, name))
            if (
                not values
                or len(set(values)) != len(values)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    or value >= len(paths)
                    for value in values
                )
            ):
                raise Lfm2ProducerFirstFoldError(
                    f"{name} must contain unique in-range row indices."
                )
            object.__setattr__(self, name, values)
        if (
            set(self.selection_indices) & set(self.fit_indices)
            or set(self.selection_indices) & set(self.validation_indices)
            or set(self.fit_indices) & set(self.validation_indices)
        ):
            raise Lfm2ProducerFirstFoldError(
                "Selection, fit, and validation rows must be disjoint."
            )
        if not callable(self.capture_child):
            raise Lfm2ProducerFirstFoldError("capture_child must be callable.")
        object.__setattr__(
            self,
            "parent_cache_fingerprint",
            _digest(self.parent_cache_fingerprint, "parent_cache_fingerprint"),
        )
        parent_rows_cache_key = _digest(
            getattr(self.parent_rows, "cache_key", None),
            "parent_rows.cache_key",
        )
        if parent_rows_cache_key != self.parent_cache_fingerprint:
            raise Lfm2ProducerFirstFoldError(
                "parent_rows.cache_key must exactly match "
                "parent_cache_fingerprint."
            )
        object.__setattr__(
            self,
            "parent_identity_sha256",
            _digest(self.parent_identity_sha256, "parent_identity_sha256"),
        )
        if not isinstance(self.capture_builder_id, str) or not self.capture_builder_id:
            raise Lfm2ProducerFirstFoldError(
                "capture_builder_id must be a nonempty string."
            )
        object.__setattr__(
            self,
            "parent_rows_identity_sha256",
            _rows_identity_sha256(self.parent_rows),
        )
        object.__setattr__(
            self,
            "evidence_sha256",
            _canonical_sha256(self.identity()),
        )

    def identity(self) -> dict[str, Any]:
        paths, splits, corpus, trace = _rows_metadata(self.parent_rows)
        return {
            "schema": PRODUCER_FIRST_EVIDENCE_SCHEMA,
            "parent_cache_fingerprint": self.parent_cache_fingerprint,
            "parent_cache_key": self.parent_cache_fingerprint,
            "parent_identity_sha256": self.parent_identity_sha256,
            "parent_rows_identity_sha256": self.parent_rows_identity_sha256,
            "parent_trace_rows_sha256": trace,
            "corpus_fingerprint": corpus,
            "ordered_paths_sha256": _canonical_sha256(list(paths)),
            "ordered_splits_sha256": _canonical_sha256(list(splits)),
            "selection_indices": list(self.selection_indices),
            "fit_indices": list(self.fit_indices),
            "validation_indices": list(self.validation_indices),
            "capture_builder_id": self.capture_builder_id,
            "blank_child_features_included": False,
            "parent_forward_count": 0,
        }

    @property
    def fingerprint(self) -> str:
        return self.evidence_sha256

    def verify(self) -> "Lfm2ProducerFirstEvidence":
        """Recompute the sealed row and metadata identities in place."""

        if _rows_identity_sha256(self.parent_rows) != (
            self.parent_rows_identity_sha256
        ):
            raise Lfm2ProducerFirstFoldError(
                "The sealed parent trace rows changed after evidence construction."
            )
        if _canonical_sha256(self.identity()) != self.evidence_sha256:
            raise Lfm2ProducerFirstFoldError(
                "Producer-first evidence metadata changed after sealing."
            )
        return self


def _authenticate_parent_evidence(
    parent: Any,
    evidence: Lfm2ProducerFirstEvidence,
) -> str:
    """Revalidate the sealed cache rows and exact live parent identity."""

    parent_rows_cache_key = _digest(
        getattr(evidence.parent_rows, "cache_key", None),
        "parent_rows.cache_key",
    )
    if parent_rows_cache_key != evidence.parent_cache_fingerprint:
        raise Lfm2ProducerFirstFoldError(
            "The sealed parent rows/cache identity changed after evidence "
            "construction."
        )
    evidence.verify()
    live_parent_identity = _live_parent_identity_sha256(parent)
    if live_parent_identity != evidence.parent_identity_sha256:
        raise Lfm2ProducerFirstFoldError(
            "The sealed trace cache is stale or belongs to a different exact "
            "loaded LFM2 parent."
        )
    return live_parent_identity


def authenticate_lfm2_producer_first_evidence(
    parent: Any,
    evidence: Lfm2ProducerFirstEvidence,
) -> str:
    """Revalidate sealed parent rows and the exact loaded parent.

    The hybrid depth-fold transaction uses the same parent-only evidence
    boundary as the full-depth producer-first path.  Exposing this narrow
    wrapper keeps that transaction from duplicating (or weakening) the cache
    and parent identity checks.
    """

    if not isinstance(evidence, Lfm2ProducerFirstEvidence):
        raise TypeError("evidence must be Lfm2ProducerFirstEvidence.")
    return _authenticate_parent_evidence(parent, evidence)


@dataclass(frozen=True, slots=True)
class Lfm2ProducerFirstFoldConfig:
    """Policy for the full-depth model transaction."""

    layer_config: Lfm2ClosedLoopFoldConfig = Lfm2ClosedLoopFoldConfig()
    endpoint_refinement_passes: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.layer_config, Lfm2ClosedLoopFoldConfig):
            raise Lfm2ProducerFirstFoldError(
                "layer_config must be Lfm2ClosedLoopFoldConfig."
            )
        object.__setattr__(
            self,
            "endpoint_refinement_passes",
            _positive_int(
                self.endpoint_refinement_passes,
                "endpoint_refinement_passes",
            ),
        )


@dataclass(frozen=True, slots=True)
class Lfm2ProducerPlan:
    """Real medoid producers plus consumer priors for one compact layer."""

    layer_index: int
    kind: str
    attention: GQAActivationClusterPlan | None
    mlp: GatedMLPActivationClusterPlan
    overrides: tuple[tuple[str, Any], ...]
    plan_sha256: str

    def local_state_overrides(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self.overrides))

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "kind": self.kind,
            "attention_medoids": (
                None
                if self.attention is None
                else {
                    "group": list(self.attention.group_medoids),
                    "query": list(self.attention.query_head_medoids),
                }
            ),
            "mlp_medoids": list(self.mlp.medoid_source_neurons),
            "plan_sha256": self.plan_sha256,
            "real_parent_producers": True,
            "blank_values_read": False,
        }


def _derive_producer(
    source_layer: Any,
    target_layer: Any,
    rows: Any,
    layer_index: int,
    evidence: Lfm2ProducerFirstEvidence,
    config: Lfm2ClosedLoopFoldConfig,
) -> Lfm2ProducerPlan:
    torch = _torch()
    source = inspect_lfm2_layer(source_layer)
    target = inspect_lfm2_layer(target_layer)
    if source.kind != target.kind or source.hidden_size != target.hidden_size:
        raise Lfm2ProducerFirstFoldError(
            "Producer-first folding requires matching full-depth layer kinds "
            "and hidden size."
        )
    selection = evidence.selection_indices
    gate = _index_rows(rows.field(layer_index, "mlp.gate"), selection)
    up = _index_rows(rows.field(layer_index, "mlp.up"), selection)
    post = _index_rows(rows.field(layer_index, "mlp.post_swiglu"), selection)
    source_down = source_layer.feed_forward.w2.weight.detach().to(
        device="cpu", dtype=torch.float64
    )
    contribution = post.to(torch.float64) * torch.linalg.vector_norm(
        source_down, dim=0
    ).unsqueeze(0)
    mlp = build_gated_mlp_activation_cluster_plan(
        gate,
        up,
        post,
        contribution,
        target_neurons=target.ffn_intermediate_size,
        source_neurons=source.ffn_intermediate_size,
        ridge=config.cluster_ridge,
        max_iterations=config.cluster_max_iterations,
        distance_chunk_size=config.cluster_distance_chunk_size,
    )
    overrides: dict[str, Any] = {
        "feed_forward.w1.weight": mlp.select_gate_producer(
            source_layer.feed_forward.w1.weight, axis=0
        ),
        "feed_forward.w3.weight": mlp.select_up_producer(
            source_layer.feed_forward.w3.weight, axis=0
        ),
        "feed_forward.w2.weight": mlp.aggregate_down_consumer(
            source_layer.feed_forward.w2.weight, axis=1
        ),
    }
    attention = None
    if source.kind == "full_attention":
        if not isinstance(source.operator, Lfm2AttentionContract) or not isinstance(
            target.operator, Lfm2AttentionContract
        ):
            raise Lfm2ProducerFirstFoldError(
                "Canonical LFM2 attention contracts are unavailable."
            )
        query = _index_rows(rows.field(layer_index, "attention.q"), selection)
        key = _index_rows(rows.field(layer_index, "attention.k"), selection)
        value = _index_rows(rows.field(layer_index, "attention.v"), selection)
        context = _index_rows(
            rows.field(layer_index, "attention.context"), selection
        )
        contributions = _attention_contributions(
            context,
            source_layer.self_attn.out_proj.weight,
            source.operator.query_heads,
            source.operator.head_dim,
        )
        attention = build_gqa_activation_cluster_plan(
            query.movedim(1, 0).contiguous(),
            key.movedim(1, 0).contiguous(),
            value.movedim(1, 0).contiguous(),
            context.movedim(1, 0).contiguous(),
            contributions.movedim(1, 0).contiguous(),
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
        overrides.update(
            {
                "self_attn.q_proj.weight": attention.select_query_producer(
                    source_layer.self_attn.q_proj.weight, axis=0
                ),
                "self_attn.k_proj.weight": attention.select_key_producer(
                    source_layer.self_attn.k_proj.weight, axis=0
                ),
                "self_attn.v_proj.weight": attention.select_value_producer(
                    source_layer.self_attn.v_proj.weight, axis=0
                ),
                "self_attn.out_proj.weight": attention.aggregate_output_consumer(
                    source_layer.self_attn.out_proj.weight, axis=1
                ),
            }
        )
    elif not isinstance(source.operator, Lfm2ShortConvContract):
        raise Lfm2ProducerFirstFoldError(
            "Canonical LFM2 short-conv contract is unavailable."
        )
    tensor_identity = {
        name: _tensor_sha256(value) for name, value in sorted(overrides.items())
    }
    body = {
        "layer_index": layer_index,
        "kind": source.kind,
        "attention_group_medoids": (
            None if attention is None else list(attention.group_medoids)
        ),
        "attention_query_medoids": (
            None if attention is None else list(attention.query_head_medoids)
        ),
        "mlp_medoids": list(mlp.medoid_source_neurons),
        "tensors": tensor_identity,
        "parent_trace_rows_sha256": getattr(rows, "trace_rows_sha256"),
        "blank_values_read": False,
    }
    return Lfm2ProducerPlan(
        layer_index=layer_index,
        kind=source.kind,
        attention=attention,
        mlp=mlp,
        overrides=tuple(
            (name, value.detach().to(device="cpu").contiguous().clone())
            for name, value in sorted(overrides.items())
        ),
        plan_sha256=_canonical_sha256(body),
    )


def derive_lfm2_producer_plan_for_layers(
    source_layer: Any,
    target_layer: Any,
    rows: Any,
    source_layer_index: int,
    evidence: Lfm2ProducerFirstEvidence,
    *,
    config: Lfm2ClosedLoopFoldConfig,
) -> Lfm2ProducerPlan:
    """Derive one parent-only producer shell for an explicit layer pairing."""

    if not isinstance(evidence, Lfm2ProducerFirstEvidence):
        raise TypeError("evidence must be Lfm2ProducerFirstEvidence.")
    if not isinstance(config, Lfm2ClosedLoopFoldConfig):
        raise TypeError("config must be Lfm2ClosedLoopFoldConfig.")
    return _derive_producer(
        source_layer,
        target_layer,
        rows,
        source_layer_index,
        evidence,
        config,
    )


def derive_lfm2_producer_plans(
    parent: Any,
    child: Any,
    evidence: Lfm2ProducerFirstEvidence,
    *,
    config: Lfm2ClosedLoopFoldConfig,
) -> tuple[Lfm2ProducerPlan, ...]:
    """Derive every producer before any child value is read or captured."""

    _authenticate_parent_evidence(parent, evidence)
    source_layers = _layers(parent)
    target_layers = _layers(child)
    parent_kinds = tuple(
        inspect_lfm2_layer(layer).kind for layer in source_layers
    )
    target_kinds = tuple(
        inspect_lfm2_layer(layer).kind for layer in target_layers
    )
    row_kinds = tuple(str(value) for value in evidence.parent_rows.layer_kinds)
    if (
        len(source_layers) != len(target_layers)
        or parent_kinds != target_kinds
        or row_kinds != parent_kinds
    ):
        raise Lfm2ProducerFirstFoldError(
            "Producer-first folding is full-depth only and requires identical "
            "ordered parent/child/trace layer kinds."
        )
    return tuple(
        _derive_producer(
            source,
            target,
            evidence.parent_rows,
            index,
            evidence,
            config,
        )
        for index, (source, target) in enumerate(
            zip(source_layers, target_layers, strict=True)
        )
    )


def _install_complete_producer_state(
    parent: Any,
    child: Any,
    plans: Sequence[Lfm2ProducerPlan],
) -> dict[str, Any]:
    torch = _torch()
    parent_state = parent.state_dict()
    child_state = child.state_dict()
    if set(parent_state) != set(child_state):
        raise Lfm2ProducerFirstFoldError(
            "Full-depth parent and child state keys differ."
        )
    prefixes = _layer_prefixes(child)
    overrides: dict[str, Any] = {}
    for plan in plans:
        prefix = prefixes[plan.layer_index]
        overrides.update(
            {
                f"{prefix}.{name}": value
                for name, value in plan.local_state_overrides().items()
            }
        )
    same_shape = 0
    folded = 0
    with torch.no_grad():
        for name, target in child_state.items():
            if name in overrides:
                source = overrides[name]
                folded += 1
            else:
                source = parent_state[name]
                same_shape += 1
            if tuple(source.shape) != tuple(target.shape):
                raise Lfm2ProducerFirstFoldError(
                    f"No complete parent-derived producer exists for {name!r}."
                )
            target.copy_(source.to(device=target.device, dtype=target.dtype))
    if any(parameter.grad is not None for parameter in child.parameters()):
        raise Lfm2ProducerFirstFoldError(
            "Producer installation unexpectedly wrote gradients."
        )
    return {
        "same_shape_parent_tensors": same_shape,
        "folded_parent_producer_tensors": folded,
        "complete_target_state_coverage": True,
        "blank_values_read": False,
        "state_sha256": _state_sha256(child),
    }


def _build_layer_evidence(
    parent_rows: Any,
    child_rows: Any,
    layer_index: int,
    evidence: Lfm2ProducerFirstEvidence,
) -> Lfm2ClosedLoopFoldEvidence:
    parent_paths, parent_splits, parent_corpus, _ = _rows_metadata(parent_rows)
    child_paths, child_splits, child_corpus, _ = _rows_metadata(child_rows)
    if (
        child_paths != parent_paths
        or child_splits != parent_splits
        or child_corpus != parent_corpus
    ):
        raise Lfm2ProducerFirstFoldError(
            "Parent and child captures are not path/corpus aligned."
        )
    if tuple(child_rows.layer_kinds) != tuple(parent_rows.layer_kinds):
        raise Lfm2ProducerFirstFoldError(
            "The captured child layer topology differs from the parent."
        )
    selection = evidence.selection_indices
    fit = evidence.fit_indices
    validation = evidence.validation_indices
    mlp = Lfm2MLPFoldEvidence(
        parent_gate_rows=_index_rows(
            parent_rows.field(layer_index, "mlp.gate"), selection
        ),
        parent_up_rows=_index_rows(
            parent_rows.field(layer_index, "mlp.up"), selection
        ),
        parent_post_swiglu_rows=_index_rows(
            parent_rows.field(layer_index, "mlp.post_swiglu"), selection
        ),
        parent_contribution_rows=None,
        child_post_swiglu_rows=_index_rows(
            child_rows.field(layer_index, "mlp.post_swiglu"), fit
        ),
        parent_branch_output_rows=_index_rows(
            parent_rows.field(layer_index, "residual.post_layer"), fit
        ),
        child_pre_branch_residual_rows=_index_rows(
            child_rows.field(layer_index, "residual.post_operator"), fit
        ),
        validation_child_post_swiglu_rows=_optional_rows(
            child_rows.field(layer_index, "mlp.post_swiglu"), validation
        ),
        validation_parent_branch_output_rows=_optional_rows(
            parent_rows.field(layer_index, "residual.post_layer"), validation
        ),
        validation_child_pre_branch_residual_rows=_optional_rows(
            child_rows.field(layer_index, "residual.post_operator"), validation
        ),
    )
    attention = None
    if parent_rows.layer_kinds[layer_index] == "full_attention":
        attention = Lfm2AttentionFoldEvidence(
            parent_query_rows=_index_rows(
                parent_rows.field(layer_index, "attention.q"), selection
            ),
            parent_key_rows=_index_rows(
                parent_rows.field(layer_index, "attention.k"), selection
            ),
            parent_value_rows=_index_rows(
                parent_rows.field(layer_index, "attention.v"), selection
            ),
            parent_context_rows=_index_rows(
                parent_rows.field(layer_index, "attention.context"), selection
            ),
            parent_contribution_rows=None,
            child_context_rows=_index_rows(
                child_rows.field(layer_index, "attention.context"), fit
            ),
            parent_branch_output_rows=_index_rows(
                parent_rows.field(layer_index, "residual.post_operator"), fit
            ),
            child_pre_branch_residual_rows=_index_rows(
                child_rows.field(layer_index, "residual.pre"), fit
            ),
            validation_child_context_rows=_optional_rows(
                child_rows.field(layer_index, "attention.context"), validation
            ),
            validation_parent_branch_output_rows=_optional_rows(
                parent_rows.field(layer_index, "residual.post_operator"), validation
            ),
            validation_child_pre_branch_residual_rows=_optional_rows(
                child_rows.field(layer_index, "residual.pre"), validation
            ),
        )
    return Lfm2ClosedLoopFoldEvidence(mlp=mlp, attention=attention)


def _assert_same_producers(
    candidate: Lfm2ClosedLoopFoldPlan,
    producer: Lfm2ProducerPlan,
) -> None:
    if (
        tuple(candidate.mlp.cluster_plan.medoid_source_neurons)
        != tuple(producer.mlp.medoid_source_neurons)
        or tuple(candidate.mlp.cluster_plan.source_to_target)
        != tuple(producer.mlp.source_to_target)
    ):
        raise Lfm2ProducerFirstFoldError(
            "Closed-loop MLP planning drifted from the sealed parent medoids."
        )
    if producer.attention is None:
        if candidate.attention is not None:
            raise Lfm2ProducerFirstFoldError(
                "A short-conv producer unexpectedly acquired attention."
            )
        return
    if candidate.attention is None:
        raise Lfm2ProducerFirstFoldError(
            "An attention producer lost its coupled attention fold."
        )
    observed = candidate.attention.cluster_plan
    if (
        tuple(observed.group_medoids) != tuple(producer.attention.group_medoids)
        or tuple(observed.group_assignment)
        != tuple(producer.attention.group_assignment)
        or tuple(observed.query_head_medoids)
        != tuple(producer.attention.query_head_medoids)
        or tuple(observed.query_head_assignment)
        != tuple(producer.attention.query_head_assignment)
    ):
        raise Lfm2ProducerFirstFoldError(
            "Closed-loop attention planning drifted from sealed parent medoids."
        )


def _capture_installed_child(
    child: Any,
    evidence: Lfm2ProducerFirstEvidence,
    *,
    forbidden_blank_state_sha256: str,
) -> tuple[Lfm2ProducerFirstCapture, str]:
    before = _state_sha256(child)
    if before == forbidden_blank_state_sha256:
        raise Lfm2ProducerFirstFoldError(
            "Refusing to capture features from the compiler-zero blank child."
        )
    capture = evidence.capture_child(child)
    if not isinstance(capture, Lfm2ProducerFirstCapture):
        raise Lfm2ProducerFirstFoldError(
            "capture_child must return Lfm2ProducerFirstCapture."
        )
    after = _state_sha256(child)
    if after != before:
        raise Lfm2ProducerFirstFoldError(
            "Child capture mutated the installed candidate state."
        )
    return capture, before


@dataclass(frozen=True, slots=True)
class Lfm2ProducerFirstModelFoldPlan:
    """Authenticated report for an already-installed whole-model candidate."""

    blank_state_sha256: str
    producer_state_sha256: str
    candidate_state_sha256: str
    producer_bundle_sha256: str
    evidence_fingerprint: str
    producer_install_report: Mapping[str, Any]
    layer_reports: tuple[Mapping[str, Any], ...]
    capture_reports: tuple[Mapping[str, Any], ...]
    child_forward_count: int
    zero_gradient: bool = True
    seed_erasure_by_construction: bool = True
    blank_features_used: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": PRODUCER_FIRST_SCHEMA,
            "method_id": PRODUCER_FIRST_MODEL_METHOD,
            "blank_state_sha256": self.blank_state_sha256,
            "producer_state_sha256": self.producer_state_sha256,
            "candidate_state_sha256": self.candidate_state_sha256,
            "producer_bundle_sha256": self.producer_bundle_sha256,
            "evidence_fingerprint": self.evidence_fingerprint,
            "producer_install_report": dict(self.producer_install_report),
            "layer_reports": [dict(value) for value in self.layer_reports],
            "capture_reports": [dict(value) for value in self.capture_reports],
            "child_forward_count": self.child_forward_count,
            "parent_forward_count": 0,
            "zero_gradient": self.zero_gradient,
            "seed_erasure_by_construction": self.seed_erasure_by_construction,
            "blank_features_used": self.blank_features_used,
            "full_depth_only": True,
            "optimizer_steps": 0,
            "backward_calls": 0,
        }


class Lfm2ProducerFirstModelTransaction:
    """One exact whole-model commit/rollback boundary."""

    method_id = PRODUCER_FIRST_MODEL_METHOD

    def __init__(
        self,
        model: Any,
        snapshot: Mapping[str, Any],
        plan: Lfm2ProducerFirstModelFoldPlan,
    ) -> None:
        self.model = model
        self.snapshot = snapshot
        self.plan = plan
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    def commit(self) -> Mapping[str, Any]:
        if not self._active:
            raise Lfm2ProducerFirstFoldError(
                "Producer-first transaction is no longer active."
            )
        if _state_sha256(self.model) != self.plan.candidate_state_sha256:
            raise Lfm2ProducerFirstFoldError(
                "Producer-first candidate changed before commit."
            )
        self._active = False
        return {
            **self.plan.as_dict(),
            "committed": True,
            "rollback_exact": True,
        }

    def rollback(self) -> Mapping[str, Any]:
        if not self._active:
            raise Lfm2ProducerFirstFoldError(
                "Producer-first transaction is no longer active."
            )
        _restore_state(self.model, self.snapshot)
        if hasattr(self.model, "_neural_forge_producer_plan_fingerprint"):
            delattr(self.model, "_neural_forge_producer_plan_fingerprint")
        restored = _state_sha256(self.model)
        if restored != self.plan.blank_state_sha256:
            raise Lfm2ProducerFirstFoldError(
                "Producer-first rollback did not exactly restore the blank."
            )
        self._active = False
        return {
            **self.plan.as_dict(),
            "committed": False,
            "rollback_exact": True,
            "restored_state_sha256": restored,
        }


def install_lfm2_producer_first_model_fold(
    parent: Any,
    child: Any,
    evidence: Lfm2ProducerFirstEvidence,
    *,
    config: Lfm2ProducerFirstFoldConfig | None = None,
) -> Lfm2ProducerFirstModelTransaction:
    """Build and install the producer-first candidate as one transaction."""

    if not isinstance(evidence, Lfm2ProducerFirstEvidence):
        raise TypeError("evidence must be Lfm2ProducerFirstEvidence.")
    selected = config or Lfm2ProducerFirstFoldConfig()
    if not isinstance(selected, Lfm2ProducerFirstFoldConfig):
        raise TypeError("config must be Lfm2ProducerFirstFoldConfig or None.")
    if parent is child:
        raise Lfm2ProducerFirstFoldError("Parent and child must be distinct models.")
    if hasattr(child, "_neural_forge_producer_plan_fingerprint"):
        raise Lfm2ProducerFirstFoldError(
            "Blank child already carries a producer-plan runtime marker."
        )
    snapshot = _clone_state(child)
    blank_state = _state_sha256(child)
    gradient_before = tuple(
        (name, parameter.grad)
        for name, parameter in child.named_parameters()
    )
    child_mutated = False
    try:
        plans = derive_lfm2_producer_plans(
            parent,
            child,
            evidence,
            config=selected.layer_config,
        )
        bundle = _canonical_sha256([plan.as_dict() for plan in plans])
        # Producer derivation can be expensive.  Close the resulting TOCTOU
        # window by re-authenticating both the rows and exact live parent
        # immediately before the first possible child write.
        _authenticate_parent_evidence(parent, evidence)
        child_mutated = True
        producer_install = _install_complete_producer_state(parent, child, plans)
        producer_state = _state_sha256(child)
        if producer_state == blank_state:
            raise Lfm2ProducerFirstFoldError(
                "Complete producer installation left the blank unchanged."
            )
        setattr(child, "_neural_forge_producer_plan_fingerprint", bundle)
        capture, captured_state = _capture_installed_child(
            child,
            evidence,
            forbidden_blank_state_sha256=blank_state,
        )
        if captured_state != producer_state:
            raise Lfm2ProducerFirstFoldError(
                "Initial child features were not captured from the authenticated "
                "producer state."
            )
        capture_reports: list[Mapping[str, Any]] = [
            MappingProxyType(
                {
                    "kind": "complete_producer_state",
                    "state_sha256": captured_state,
                    "cache_key": capture.cache_key,
                    "trace_rows_sha256": capture.rows.trace_rows_sha256,
                    "child_forward_count": capture.child_forward_count,
                    "cache_reused": capture.cache_reused,
                }
            )
        ]
        child_forward_count = capture.child_forward_count
        current_rows = capture.rows
        layer_reports: list[Mapping[str, Any]] = []
        source_layers = _layers(parent)
        target_layers = _layers(child)
        for producer in plans:
            source_layer = source_layers[producer.layer_index]
            target_layer = target_layers[producer.layer_index]
            passes: list[Mapping[str, Any]] = []
            for pass_index in range(selected.endpoint_refinement_passes):
                installed_feature_state = _state_sha256(child)
                layer_evidence = _build_layer_evidence(
                    evidence.parent_rows,
                    current_rows,
                    producer.layer_index,
                    evidence,
                )
                plan = build_lfm2_closed_loop_fold_plan(
                    source_layer,
                    target_layer,
                    layer_evidence,
                    config=selected.layer_config,
                )
                _assert_same_producers(plan, producer)
                layer_transaction = install_lfm2_closed_loop_fold_plan(
                    target_layer,
                    plan,
                )
                try:
                    capture, captured_candidate_state = _capture_installed_child(
                        child,
                        evidence,
                        forbidden_blank_state_sha256=blank_state,
                    )
                    layer_transaction.commit()
                except BaseException:
                    if layer_transaction.active:
                        layer_transaction.rollback()
                    raise
                child_forward_count += capture.child_forward_count
                current_rows = capture.rows
                capture_reports.append(
                    MappingProxyType(
                        {
                            "kind": "accepted_layer_refinement",
                            "layer_index": producer.layer_index,
                            "pass_index": pass_index,
                            "state_sha256": captured_candidate_state,
                            "cache_key": capture.cache_key,
                            "trace_rows_sha256": capture.rows.trace_rows_sha256,
                            "child_forward_count": capture.child_forward_count,
                            "cache_reused": capture.cache_reused,
                        }
                    )
                )
                passes.append(
                    MappingProxyType(
                        {
                            "pass_index": pass_index,
                            "installed_feature_state_sha256": (
                                installed_feature_state
                            ),
                            "candidate_state_sha256": captured_candidate_state,
                            "candidate_features_recaptured": True,
                            "plan_sha256": plan.plan_sha256,
                        }
                    )
                )
            layer_reports.append(
                MappingProxyType(
                    {
                        "layer_index": producer.layer_index,
                        "kind": producer.kind,
                        "producer_plan_sha256": producer.plan_sha256,
                        "passes": [dict(value) for value in passes],
                        "accepted": True,
                    }
                )
            )
        candidate_state = _state_sha256(child)
        if candidate_state == blank_state:
            raise Lfm2ProducerFirstFoldError(
                "Producer-first candidate regressed to the blank state."
            )
        if (
            _live_parent_identity_sha256(parent)
            != evidence.parent_identity_sha256
        ):
            raise Lfm2ProducerFirstFoldError(
                "Producer-first folding changed the exact parent identity."
            )
        if tuple(
            (name, parameter.grad)
            for name, parameter in child.named_parameters()
        ) != gradient_before:
            raise Lfm2ProducerFirstFoldError(
                "Producer-first folding changed parameter gradients."
            )
        plan = Lfm2ProducerFirstModelFoldPlan(
            blank_state_sha256=blank_state,
            producer_state_sha256=producer_state,
            candidate_state_sha256=candidate_state,
            producer_bundle_sha256=bundle,
            evidence_fingerprint=evidence.fingerprint,
            producer_install_report=MappingProxyType(dict(producer_install)),
            layer_reports=tuple(layer_reports),
            capture_reports=tuple(capture_reports),
            child_forward_count=child_forward_count,
        )
        return Lfm2ProducerFirstModelTransaction(child, snapshot, plan)
    except BaseException:
        if hasattr(child, "_neural_forge_producer_plan_fingerprint"):
            delattr(child, "_neural_forge_producer_plan_fingerprint")
        if child_mutated:
            _restore_state(child, snapshot)
        if _state_sha256(child) != blank_state:
            raise Lfm2ProducerFirstFoldError(
                "Producer-first failure could not restore the blank exactly."
            )
        raise


__all__ = [
    "Lfm2ProducerFirstCapture",
    "Lfm2ProducerFirstEvidence",
    "Lfm2ProducerFirstFoldConfig",
    "Lfm2ProducerFirstFoldError",
    "Lfm2ProducerFirstModelFoldPlan",
    "Lfm2ProducerFirstModelTransaction",
    "PRODUCER_FIRST_MODEL_METHOD",
    "authenticate_lfm2_producer_first_evidence",
    "derive_lfm2_producer_plan_for_layers",
    "derive_lfm2_producer_plans",
    "install_lfm2_producer_first_model_fold",
]
