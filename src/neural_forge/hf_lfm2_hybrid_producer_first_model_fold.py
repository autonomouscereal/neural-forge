"""Producer-first staging for atomic LFM2 depth+width model folding.

The hybrid model fold needs child-side activation designs, but those designs
must never come from the compiler-zero child.  This module owns the missing
transactional sequence:

1. authenticate sealed parent trace rows without executing either model;
2. choose one topology-compatible real parent layer in every ordered span;
3. install a complete parent-derived producer shell across the compact child;
4. capture only that installed shell through the bound child callback;
5. assemble :class:`Lfm2ModelFoldEvidence` from the sealed/captured rows;
6. build and install the final residual-attention model plan; and
7. expose one commit/byte-exact-rollback boundary whose incumbent is the
   original compiler-zero child, not the provisional producer shell.

No parent forward, optimizer, backward call, gradient write, or blank-child
feature capture is performed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .hf_lfm2_canonical import inspect_lfm2_layer
from .hf_lfm2_model_fold import (
    Lfm2ModelFoldConfig,
    Lfm2ModelFoldPlan,
    Lfm2ModelFoldSpan,
    Lfm2ModelSnapshot,
    assemble_lfm2_model_fold_evidence_from_trace_rows,
    build_lfm2_model_fold_plan,
    fingerprint_lfm2_model_feature_state,
    install_lfm2_model_fold_plan,
    restore_lfm2_model_snapshot,
    snapshot_lfm2_model,
)
from .hf_lfm2_producer_first_model_fold import (
    Lfm2ProducerFirstCapture,
    Lfm2ProducerFirstEvidence,
    Lfm2ProducerPlan,
    authenticate_lfm2_producer_first_evidence,
    derive_lfm2_producer_plan_for_layers,
)


HYBRID_PRODUCER_FIRST_METHOD = "lfm2_hybrid_residual_attention_model"
HYBRID_PRODUCER_FIRST_SCHEMA = (
    "neural-forge-lfm2-hybrid-producer-first-model-fold-v1"
)


class Lfm2HybridProducerFirstFoldError(RuntimeError):
    """The hybrid producer-first transaction could not prove its contract."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise Lfm2HybridProducerFirstFoldError(
            "PyTorch is required for hybrid producer-first folding."
        ) from exc
    return torch


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
        raise Lfm2HybridProducerFirstFoldError(
            "Hybrid producer provenance is not canonical JSON."
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _layers(model: Any) -> tuple[Any, ...]:
    body = getattr(model, "model", model)
    layers = getattr(body, "layers", None)
    try:
        result = tuple(layers)
    except TypeError as exc:
        raise Lfm2HybridProducerFirstFoldError(
            "Expected model.model.layers on an LFM2 causal model."
        ) from exc
    if not result:
        raise Lfm2HybridProducerFirstFoldError(
            "An LFM2 model must contain decoder layers."
        )
    return result


def _layer_prefixes(model: Any, layers: tuple[Any, ...]) -> tuple[str, ...]:
    by_identity = {id(layer): index for index, layer in enumerate(layers)}
    result: dict[int, str] = {}
    for name, module in model.named_modules():
        index = by_identity.get(id(module))
        if index is not None and name:
            result[index] = name
    if set(result) != set(range(len(layers))):
        raise Lfm2HybridProducerFirstFoldError(
            "Could not bind every compact LFM2 layer to a state path."
        )
    return tuple(result[index] for index in range(len(layers)))


def _validate_spans(
    spans: Sequence[Lfm2ModelFoldSpan],
    *,
    source_layer_count: int,
    target_layer_count: int,
) -> tuple[Lfm2ModelFoldSpan, ...]:
    try:
        result = tuple(spans)
    except TypeError as exc:
        raise Lfm2HybridProducerFirstFoldError(
            "spans must be an LFM2 model-fold span sequence."
        ) from exc
    if (
        len(result) != target_layer_count
        or any(not isinstance(value, Lfm2ModelFoldSpan) for value in result)
        or tuple(value.target_layer_index for value in result)
        != tuple(range(target_layer_count))
        or tuple(
            index
            for span in result
            for index in span.source_layer_indices
        )
        != tuple(range(source_layer_count))
    ):
        raise Lfm2HybridProducerFirstFoldError(
            "Source spans must partition every parent layer in order and bind "
            "exactly one span to every compact target layer."
        )
    return result


def _index_rows(value: Any, indices: tuple[int, ...], name: str) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim < 1
    ):
        raise Lfm2HybridProducerFirstFoldError(
            f"{name} must be a materialized dense tensor with a row axis."
        )
    index = torch.tensor(indices, dtype=torch.long, device=value.device)
    result = value.index_select(0, index).detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2HybridProducerFirstFoldError(
            f"{name} contains NaN or infinity."
        )
    return result


def _anchor_for_span(
    span: Lfm2ModelFoldSpan,
    *,
    parent_layers: tuple[Any, ...],
    target_layer: Any,
    parent_rows: Any,
    fit_indices: tuple[int, ...],
) -> tuple[int, tuple[tuple[int, float], ...]]:
    torch = _torch()
    target_kind = inspect_lfm2_layer(target_layer).kind
    span_pre = _index_rows(
        parent_rows.field(span.source_layer_indices[0], "residual.pre"),
        fit_indices,
        "span residual.pre",
    )
    span_post = _index_rows(
        parent_rows.field(
            span.source_layer_indices[-1],
            "residual.post_layer",
        ),
        fit_indices,
        "span residual.post_layer",
    )
    if span_pre.shape != span_post.shape:
        raise Lfm2HybridProducerFirstFoldError(
            "A source span has incompatible cached residual endpoints."
        )
    span_delta = span_post - span_pre
    scale = max(
        float(torch.sqrt(torch.mean(span_delta.square())).item()),
        torch.finfo(torch.float64).tiny,
    )
    scores: list[tuple[int, float]] = []
    for source_index in span.source_layer_indices:
        if inspect_lfm2_layer(parent_layers[source_index]).kind != target_kind:
            continue
        layer_pre = _index_rows(
            parent_rows.field(source_index, "residual.pre"),
            fit_indices,
            f"source layer {source_index} residual.pre",
        )
        layer_post = _index_rows(
            parent_rows.field(source_index, "residual.post_layer"),
            fit_indices,
            f"source layer {source_index} residual.post_layer",
        )
        if layer_pre.shape != span_pre.shape or layer_post.shape != span_pre.shape:
            raise Lfm2HybridProducerFirstFoldError(
                "Cached layer residuals are not aligned to their source span."
            )
        rmse = float(
            torch.sqrt(
                torch.mean((span_delta - (layer_post - layer_pre)).square())
            ).item()
        )
        relative = rmse / scale
        if not math.isfinite(relative):
            raise Lfm2HybridProducerFirstFoldError(
                "Hybrid anchor score is not finite."
            )
        scores.append((source_index, relative))
    if not scores:
        raise Lfm2HybridProducerFirstFoldError(
            "A source span has no real layer compatible with its target operator."
        )
    ordered = tuple(sorted(scores, key=lambda item: (item[1], item[0])))
    return ordered[0][0], ordered


@dataclass(frozen=True, slots=True)
class Lfm2HybridProducerBinding:
    """One real parent anchor bound to one compact target layer."""

    span: Lfm2ModelFoldSpan
    anchor_source_layer_index: int
    anchor_scores: tuple[tuple[int, float], ...]
    producer: Lfm2ProducerPlan

    def as_dict(self) -> dict[str, Any]:
        return {
            "span": self.span.as_dict(),
            "anchor_source_layer_index": self.anchor_source_layer_index,
            "anchor_scores": [
                {
                    "source_layer_index": source_index,
                    "residual_delta_relative_rmse": score,
                }
                for source_index, score in self.anchor_scores
            ],
            "producer_plan": self.producer.as_dict(),
            "real_parent_anchor": True,
            "blank_values_used_as_features": False,
        }


def derive_lfm2_hybrid_producer_bindings(
    parent: Any,
    child: Any,
    spans: Sequence[Lfm2ModelFoldSpan],
    evidence: Lfm2ProducerFirstEvidence,
    *,
    config: Lfm2ModelFoldConfig,
) -> tuple[Lfm2HybridProducerBinding, ...]:
    """Derive every span producer before the first child write."""

    if not isinstance(evidence, Lfm2ProducerFirstEvidence):
        raise TypeError("evidence must be Lfm2ProducerFirstEvidence.")
    if not isinstance(config, Lfm2ModelFoldConfig):
        raise TypeError("config must be Lfm2ModelFoldConfig.")
    authenticate_lfm2_producer_first_evidence(parent, evidence)
    parent_layers = _layers(parent)
    child_layers = _layers(child)
    span_values = _validate_spans(
        spans,
        source_layer_count=len(parent_layers),
        target_layer_count=len(child_layers),
    )
    parent_kinds = tuple(
        inspect_lfm2_layer(layer).kind for layer in parent_layers
    )
    row_kinds = tuple(
        str(value) for value in getattr(evidence.parent_rows, "layer_kinds", ())
    )
    if row_kinds != parent_kinds:
        raise Lfm2HybridProducerFirstFoldError(
            "Sealed parent trace topology differs from the loaded parent."
        )
    return tuple(
        Lfm2HybridProducerBinding(
            span=span,
            anchor_source_layer_index=anchor_index,
            anchor_scores=scores,
            producer=derive_lfm2_producer_plan_for_layers(
                parent_layers[anchor_index],
                child_layers[span.target_layer_index],
                evidence.parent_rows,
                anchor_index,
                evidence,
                config=config.layer_config,
            ),
        )
        for span in span_values
        for anchor_index, scores in (
            _anchor_for_span(
                span,
                parent_layers=parent_layers,
                target_layer=child_layers[span.target_layer_index],
                parent_rows=evidence.parent_rows,
                fit_indices=evidence.fit_indices,
            ),
        )
    )


def _complete_producer_candidate(
    parent: Any,
    child: Any,
    bindings: tuple[Lfm2HybridProducerBinding, ...],
) -> tuple[Mapping[str, Any], int, int]:
    parent_layers = _layers(parent)
    child_layers = _layers(child)
    child_prefixes = _layer_prefixes(child, child_layers)
    parent_state = parent.state_dict()
    child_state = child.state_dict()
    candidate: dict[str, Any] = {}
    folded = 0
    anchor_copies = 0
    for binding in bindings:
        target_index = binding.span.target_layer_index
        prefix = child_prefixes[target_index]
        source_local = parent_layers[
            binding.anchor_source_layer_index
        ].state_dict()
        target_local = child_layers[target_index].state_dict()
        overrides = binding.producer.local_state_overrides()
        for local_name, target in target_local.items():
            if local_name in overrides:
                source = overrides[local_name]
                folded += 1
            else:
                if local_name not in source_local:
                    raise Lfm2HybridProducerFirstFoldError(
                        "The selected parent anchor cannot cover target state "
                        f"{prefix}.{local_name}."
                    )
                source = source_local[local_name]
                anchor_copies += 1
            if tuple(source.shape) != tuple(target.shape):
                raise Lfm2HybridProducerFirstFoldError(
                    "The selected parent anchor has no shape-compatible producer "
                    f"for {prefix}.{local_name}."
                )
            candidate[f"{prefix}.{local_name}"] = (
                source.detach().to(device="cpu").contiguous().clone()
            )
    for name, target in child_state.items():
        if name in candidate:
            source = candidate[name]
        else:
            if name not in parent_state:
                raise Lfm2HybridProducerFirstFoldError(
                    f"The parent cannot cover non-layer child state {name!r}."
                )
            source = parent_state[name]
        if tuple(source.shape) != tuple(target.shape) or source.dtype != target.dtype:
            raise Lfm2HybridProducerFirstFoldError(
                f"Complete producer state is incompatible with {name!r}."
            )
        candidate[name] = source.detach().to(device="cpu").contiguous().clone()
    if set(candidate) != set(child_state):
        raise Lfm2HybridProducerFirstFoldError(
            "The provisional producer shell does not cover the whole child."
        )
    return MappingProxyType(candidate), folded, anchor_copies


def _install_candidate(child: Any, candidate: Mapping[str, Any]) -> None:
    torch = _torch()
    live = child.state_dict()
    if set(live) != set(candidate):
        raise Lfm2HybridProducerFirstFoldError(
            "The provisional producer candidate no longer matches the child."
        )
    with torch.no_grad():
        for name in sorted(live):
            target = live[name]
            source = candidate[name]
            if tuple(source.shape) != tuple(target.shape) or source.dtype != target.dtype:
                raise Lfm2HybridProducerFirstFoldError(
                    f"Producer candidate contract changed for {name!r}."
                )
            target.copy_(source.to(device=target.device))


@dataclass(frozen=True, slots=True)
class Lfm2HybridProducerFirstPlan:
    """Authenticated report for the final installed hybrid candidate."""

    blank_state_sha256: str
    producer_state_sha256: str
    candidate_state_sha256: str
    producer_bundle_sha256: str
    evidence_fingerprint: str
    bindings: tuple[Lfm2HybridProducerBinding, ...]
    model_plan: Lfm2ModelFoldPlan
    capture_cache_key: str
    capture_trace_rows_sha256: str
    child_forward_count: int
    producer_folded_tensors: int
    producer_anchor_copied_tensors: int
    zero_gradient: bool = True
    seed_erasure_by_construction: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": HYBRID_PRODUCER_FIRST_SCHEMA,
            "method_id": HYBRID_PRODUCER_FIRST_METHOD,
            "blank_state_sha256": self.blank_state_sha256,
            "producer_state_sha256": self.producer_state_sha256,
            "candidate_state_sha256": self.candidate_state_sha256,
            "producer_bundle_sha256": self.producer_bundle_sha256,
            "evidence_fingerprint": self.evidence_fingerprint,
            "producer_bindings": [value.as_dict() for value in self.bindings],
            "producer_complete_target_state_coverage": True,
            "producer_folded_tensors": self.producer_folded_tensors,
            "producer_anchor_copied_tensors": (
                self.producer_anchor_copied_tensors
            ),
            "capture": {
                "state_sha256": self.producer_state_sha256,
                "cache_key": self.capture_cache_key,
                "trace_rows_sha256": self.capture_trace_rows_sha256,
                "child_forward_count": self.child_forward_count,
                "captured_compiler_zero_blank": False,
            },
            "model_plan": self.model_plan.as_dict(),
            "child_forward_count": self.child_forward_count,
            "parent_forward_count": 0,
            "blank_features_used": False,
            "atomic_model_level_transaction": True,
            "zero_gradient": self.zero_gradient,
            "seed_erasure_by_construction": (
                self.seed_erasure_by_construction
            ),
            "optimizer_steps": 0,
            "backward_calls": 0,
            "training_steps": 0,
        }


class Lfm2HybridProducerFirstTransaction:
    """One external commit/rollback boundary rooted at the original blank."""

    method_id = HYBRID_PRODUCER_FIRST_METHOD

    def __init__(
        self,
        child: Any,
        blank_snapshot: Lfm2ModelSnapshot,
        plan: Lfm2HybridProducerFirstPlan,
    ) -> None:
        self.child = child
        self.blank_snapshot = blank_snapshot
        self.plan = plan
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    def commit(self) -> Mapping[str, Any]:
        if not self._active:
            raise Lfm2HybridProducerFirstFoldError(
                "Hybrid producer-first transaction is no longer active."
            )
        if (
            fingerprint_lfm2_model_feature_state(self.child)
            != self.plan.candidate_state_sha256
        ):
            raise Lfm2HybridProducerFirstFoldError(
                "Hybrid producer-first candidate changed before commit."
            )
        self._active = False
        return {
            **self.plan.as_dict(),
            "committed": True,
            "rollback_exact": None,
        }

    def rollback(self) -> Mapping[str, Any]:
        if not self._active:
            raise Lfm2HybridProducerFirstFoldError(
                "Hybrid producer-first transaction is no longer active."
            )
        if hasattr(self.child, "_neural_forge_producer_plan_fingerprint"):
            delattr(self.child, "_neural_forge_producer_plan_fingerprint")
        exact = restore_lfm2_model_snapshot(self.child, self.blank_snapshot)
        restored = fingerprint_lfm2_model_feature_state(self.child)
        if not exact or restored != self.plan.blank_state_sha256:
            raise Lfm2HybridProducerFirstFoldError(
                "Hybrid producer-first rollback did not restore the original "
                "blank byte-exactly."
            )
        self._active = False
        return {
            **self.plan.as_dict(),
            "committed": False,
            "rollback_exact": True,
            "restored_state_sha256": restored,
        }


def install_lfm2_hybrid_producer_first_model_fold(
    parent: Any,
    child: Any,
    spans: Sequence[Lfm2ModelFoldSpan],
    evidence: Lfm2ProducerFirstEvidence,
    *,
    config: Lfm2ModelFoldConfig | None = None,
) -> Lfm2HybridProducerFirstTransaction:
    """Stage producers, capture them, and install the final hybrid candidate."""

    if not isinstance(evidence, Lfm2ProducerFirstEvidence):
        raise TypeError("evidence must be Lfm2ProducerFirstEvidence.")
    selected = Lfm2ModelFoldConfig() if config is None else config
    if not isinstance(selected, Lfm2ModelFoldConfig):
        raise TypeError("config must be Lfm2ModelFoldConfig or None.")
    if parent is child:
        raise Lfm2HybridProducerFirstFoldError(
            "Parent and child must be distinct models."
        )
    if hasattr(child, "_neural_forge_producer_plan_fingerprint"):
        raise Lfm2HybridProducerFirstFoldError(
            "Compiler-zero child already carries a producer-plan marker."
        )
    parent_identity = authenticate_lfm2_producer_first_evidence(
        parent,
        evidence,
    )
    blank_snapshot = snapshot_lfm2_model(child)
    blank_state = fingerprint_lfm2_model_feature_state(child)
    if blank_state != blank_snapshot.state_sha256:
        raise Lfm2HybridProducerFirstFoldError(
            "The blank snapshot and live child state fingerprints differ."
        )
    bindings = derive_lfm2_hybrid_producer_bindings(
        parent,
        child,
        spans,
        evidence,
        config=selected,
    )
    span_values = tuple(binding.span for binding in bindings)
    bundle = _canonical_sha256(
        {
            "schema": HYBRID_PRODUCER_FIRST_SCHEMA,
            "evidence_fingerprint": evidence.fingerprint,
            "bindings": [value.as_dict() for value in bindings],
        }
    )
    candidate, folded_tensors, anchor_copies = _complete_producer_candidate(
        parent,
        child,
        bindings,
    )
    child_mutated = False
    inner_transaction = None
    try:
        # Close the derivation/write TOCTOU window immediately before mutation.
        if (
            authenticate_lfm2_producer_first_evidence(parent, evidence)
            != parent_identity
        ):
            raise Lfm2HybridProducerFirstFoldError(
                "The loaded parent identity changed before producer install."
            )
        child_mutated = True
        _install_candidate(child, candidate)
        producer_state = fingerprint_lfm2_model_feature_state(child)
        if producer_state == blank_state:
            raise Lfm2HybridProducerFirstFoldError(
                "Complete producer installation left the blank unchanged."
            )
        setattr(child, "_neural_forge_producer_plan_fingerprint", bundle)
        before_capture = fingerprint_lfm2_model_feature_state(child)
        capture = evidence.capture_child(child)
        if not isinstance(capture, Lfm2ProducerFirstCapture):
            raise Lfm2HybridProducerFirstFoldError(
                "capture_child must return Lfm2ProducerFirstCapture."
            )
        after_capture = fingerprint_lfm2_model_feature_state(child)
        if (
            before_capture != producer_state
            or after_capture != producer_state
            or after_capture == blank_state
        ):
            raise Lfm2HybridProducerFirstFoldError(
                "Hybrid child features were not captured from the exact "
                "installed producer shell."
            )
        model_evidence = (
            assemble_lfm2_model_fold_evidence_from_trace_rows(
                cache_fingerprint=evidence.parent_cache_fingerprint,
                child_feature_state_sha256=producer_state,
                spans=span_values,
                parent_rows=evidence.parent_rows,
                child_rows=capture.rows,
                selection_indices=evidence.selection_indices,
                fit_indices=evidence.fit_indices,
                validation_indices=evidence.validation_indices,
            )
        )
        model_plan = build_lfm2_model_fold_plan(
            parent,
            child,
            span_values,
            model_evidence,
            config=selected,
        )
        inner_transaction = install_lfm2_model_fold_plan(child, model_plan)
        inner_transaction.commit()
        candidate_state = fingerprint_lfm2_model_feature_state(child)
        if candidate_state != model_plan.candidate_state_sha256:
            raise Lfm2HybridProducerFirstFoldError(
                "Installed hybrid candidate differs from its final model plan."
            )
        if (
            authenticate_lfm2_producer_first_evidence(parent, evidence)
            != parent_identity
        ):
            raise Lfm2HybridProducerFirstFoldError(
                "Hybrid producer-first folding changed the loaded parent."
            )
        plan = Lfm2HybridProducerFirstPlan(
            blank_state_sha256=blank_state,
            producer_state_sha256=producer_state,
            candidate_state_sha256=candidate_state,
            producer_bundle_sha256=bundle,
            evidence_fingerprint=evidence.fingerprint,
            bindings=bindings,
            model_plan=model_plan,
            capture_cache_key=capture.cache_key,
            capture_trace_rows_sha256=str(capture.rows.trace_rows_sha256),
            child_forward_count=capture.child_forward_count,
            producer_folded_tensors=folded_tensors,
            producer_anchor_copied_tensors=anchor_copies,
        )
        return Lfm2HybridProducerFirstTransaction(
            child,
            blank_snapshot,
            plan,
        )
    except BaseException:
        if inner_transaction is not None and inner_transaction.active:
            inner_transaction.rollback()
        if hasattr(child, "_neural_forge_producer_plan_fingerprint"):
            delattr(child, "_neural_forge_producer_plan_fingerprint")
        if child_mutated:
            restored = restore_lfm2_model_snapshot(child, blank_snapshot)
            if not restored:
                raise Lfm2HybridProducerFirstFoldError(
                    "Hybrid producer-first failure could not restore the blank."
                )
        if fingerprint_lfm2_model_feature_state(child) != blank_state:
            raise Lfm2HybridProducerFirstFoldError(
                "Hybrid producer-first failure did not restore the blank exactly."
            )
        raise


__all__ = [
    "HYBRID_PRODUCER_FIRST_METHOD",
    "HYBRID_PRODUCER_FIRST_SCHEMA",
    "Lfm2HybridProducerBinding",
    "Lfm2HybridProducerFirstFoldError",
    "Lfm2HybridProducerFirstPlan",
    "Lfm2HybridProducerFirstTransaction",
    "derive_lfm2_hybrid_producer_bindings",
    "install_lfm2_hybrid_producer_first_model_fold",
]
