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
    MODEL_FOLD_TENSOR_HASH_SCHEME,
    Lfm2ModelFoldConfig,
    Lfm2ModelFoldPlan,
    Lfm2ModelFoldSpan,
    Lfm2ModelSnapshot,
    assemble_lfm2_model_fold_evidence_from_trace_rows,
    build_lfm2_model_fold_plan,
    fingerprint_lfm2_model_fold_tensor,
    fingerprint_lfm2_model_feature_state,
    install_lfm2_model_fold_plan,
    overlay_lfm2_span_coupled_candidates,
    restore_lfm2_model_snapshot,
    snapshot_lfm2_model,
)
from .hf_lfm2_producer_first_model_fold import (
    Lfm2ProducerFirstCapture,
    Lfm2ProducerFirstEvidence,
    Lfm2ProducerPlan,
    _rows_identity_sha256,
    authenticate_lfm2_producer_first_evidence,
    derive_lfm2_producer_plan_for_layers,
)


HYBRID_PRODUCER_FIRST_METHOD = "lfm2_hybrid_residual_attention_model"
HYBRID_PRODUCER_FIRST_SCHEMA = "neural-forge-lfm2-hybrid-producer-first-model-fold-v1"
_SPAN_COUPLED_PRODUCTION_RECEIPT_V1 = (
    "neural-forge-lfm2-span-coupled-production-receipt-v1"
)
_SPAN_COUPLED_PRODUCTION_RECEIPT_V2 = (
    "neural-forge-lfm2-span-coupled-production-receipt-v2"
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
        or tuple(index for span in result for index in span.source_layer_indices)
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
    result = (
        value.index_select(0, index)
        .detach()
        .to(
            device="cpu",
            dtype=torch.float64,
        )
        .contiguous()
    )
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2HybridProducerFirstFoldError(f"{name} contains NaN or infinity.")
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
            raise Lfm2HybridProducerFirstFoldError("Hybrid anchor score is not finite.")
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
    parent_kinds = tuple(inspect_lfm2_layer(layer).kind for layer in parent_layers)
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
        source_local = parent_layers[binding.anchor_source_layer_index].state_dict()
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
            if (
                tuple(source.shape) != tuple(target.shape)
                or source.dtype != target.dtype
            ):
                raise Lfm2HybridProducerFirstFoldError(
                    f"Producer candidate contract changed for {name!r}."
                )
            target.copy_(source.to(device=target.device))


def _authenticated_tensor_sha256(value: Any) -> str:
    from .hf_lfm2_structural_residual_attention_fold import (
        lfm2_authenticated_tensor_sha256,
    )

    return lfm2_authenticated_tensor_sha256(value)


def _norm_epsilon(model: Any, label: str) -> float:
    try:
        value = float(getattr(getattr(model, "config", None), "norm_eps"))
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise Lfm2HybridProducerFirstFoldError(
            f"{label} must expose one finite positive config.norm_eps."
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise Lfm2HybridProducerFirstFoldError(
            f"{label} config.norm_eps must be finite and positive."
        )
    return value


def _layer_norm_weight(layer: Any, name: str) -> Any:
    module = getattr(layer, name, None)
    weight = getattr(module, "weight", None)
    torch = _torch()
    if (
        not isinstance(weight, torch.Tensor)
        or weight.layout != torch.strided
        or weight.device.type == "meta"
        or weight.ndim != 1
    ):
        raise Lfm2HybridProducerFirstFoldError(
            f"LFM2 layer {name}.weight is not one materialized dense vector."
        )
    return weight


def _compile_normalized_span_decoder(
    *,
    parent_model: Any,
    child_model: Any,
    parent_layer: Any,
    target_layer: Any,
    parent_rows: Any,
    child_rows: Any,
    source_layer_index: int,
    target_layer_index: int,
    fit_indices: tuple[int, ...],
    heldout_indices: tuple[int, ...],
    role: str,
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Reuse the semantic stack's chart/gamma normalized decoder algebra."""

    from .hf_lfm2_cpcf_attention_dimension_fold import lfm2_runtime_rms_norm
    from .hf_lfm2_semantic_postfold import (
        _fit_normalized_operator_input_decoder,
    )

    if role == "mlp":
        parent_field = "residual.post_operator"
        child_field = "residual.post_operator"
        parent_norm = _layer_norm_weight(parent_layer, "ffn_norm")
        target_norm = _layer_norm_weight(target_layer, "ffn_norm")
    elif role == "attention":
        parent_field = "residual.pre"
        child_field = "residual.pre"
        parent_norm = _layer_norm_weight(parent_layer, "operator_norm")
        target_norm = _layer_norm_weight(target_layer, "operator_norm")
    else:  # pragma: no cover - internal call contract
        raise Lfm2HybridProducerFirstFoldError(
            f"Unsupported span decoder role {role!r}."
        )
    parent_fit = _index_rows(
        parent_rows.field(source_layer_index, parent_field),
        fit_indices,
        f"source layer {source_layer_index} {role} FIT input",
    )
    parent_heldout = _index_rows(
        parent_rows.field(source_layer_index, parent_field),
        heldout_indices,
        f"source layer {source_layer_index} {role} held-out input",
    )
    child_fit_raw = _index_rows(
        child_rows.field(target_layer_index, child_field),
        fit_indices,
        f"target layer {target_layer_index} {role} FIT input",
    )
    child_heldout_raw = _index_rows(
        child_rows.field(target_layer_index, child_field),
        heldout_indices,
        f"target layer {target_layer_index} {role} held-out input",
    )
    source_hidden = int(parent_fit.shape[1])
    target_hidden = int(child_fit_raw.shape[1])
    if source_hidden != target_hidden:
        raise Lfm2HybridProducerFirstFoldError(
            "The hybrid producer-first evidence has no authenticated cross-width "
            "residual chart; span-coupled compilation currently requires the "
            "shared LFM2 residual width."
        )
    torch = _torch()
    chart = torch.eye(source_hidden, dtype=torch.float64)
    chart_sha256 = _authenticated_tensor_sha256(chart)
    target_epsilon = _norm_epsilon(child_model, "child")
    child_fit = lfm2_runtime_rms_norm(
        child_fit_raw,
        target_norm,
        target_epsilon,
    )
    child_heldout = lfm2_runtime_rms_norm(
        child_heldout_raw,
        target_norm,
        target_epsilon,
    )
    decoder, raw_receipt = _fit_normalized_operator_input_decoder(
        parent_raw_rows=(parent_fit, parent_heldout),
        child_normalized_rows=(child_fit, child_heldout),
        parent_operator_norm_weight=parent_norm,
        target_operator_norm_weight=target_norm,
        parent_norm_epsilon=_norm_epsilon(parent_model, "parent"),
        target_norm_epsilon=target_epsilon,
        residual_chart=chart,
        residual_chart_sha256=chart_sha256,
        fit_trajectory_count=1,
    )
    receipt = dict(raw_receipt)
    claimed_receipt_sha256 = receipt.pop("decoder_sha256", None)
    decoder_sha256 = _authenticated_tensor_sha256(decoder)
    if (
        claimed_receipt_sha256 != _canonical_sha256(receipt)
        or receipt.get("schema")
        != "neural-forge-lfm2-normalized-operator-input-decoder-v1"
        or receipt.get("residual_chart_sha256") != chart_sha256
        or receipt.get("residual_chart_observed_sha256") != chart_sha256
        or receipt.get("published_decoder_sha256") != decoder_sha256
        or receipt.get("training_steps") != 0
        or receipt.get("gradients_used") is not False
    ):
        raise Lfm2HybridProducerFirstFoldError(
            "The existing normalized-input decoder receipt failed authentication."
        )
    bound_receipt = {
        **receipt,
        "decoder_receipt_sha256": claimed_receipt_sha256,
        "decoder_tensor_sha256": decoder_sha256,
        "role": role,
        "source_layer_index": source_layer_index,
        "target_layer_index": target_layer_index,
        "parent_fit_rows_sha256": _authenticated_tensor_sha256(parent_fit),
        "parent_heldout_rows_sha256": _authenticated_tensor_sha256(parent_heldout),
        "child_fit_rows_sha256": _authenticated_tensor_sha256(child_fit),
        "child_heldout_rows_sha256": _authenticated_tensor_sha256(child_heldout),
        "same_hidden_identity_chart_is_architecture_exact": True,
    }
    return decoder, chart, MappingProxyType(bound_receipt)


def _span_source_receipt_sha256(
    *,
    evidence: Lfm2ProducerFirstEvidence,
    capture: Lfm2ProducerFirstCapture,
    producer_state_sha256: str,
    capture_rows_identity_sha256: str,
    span: Lfm2ModelFoldSpan,
    source_layer_index: int,
    target_layer_index: int,
    role: str,
    cached_rows_sha256: str,
    decoder_receipt: Mapping[str, Any],
) -> str:
    return _canonical_sha256(
        {
            "schema": "neural-forge-lfm2-span-source-production-binding-v1",
            "parent_identity_sha256": evidence.parent_identity_sha256,
            "parent_cache_fingerprint": evidence.parent_cache_fingerprint,
            "parent_rows_identity_sha256": evidence.parent_rows_identity_sha256,
            "producer_evidence_sha256": evidence.evidence_sha256,
            "capture_cache_key": capture.cache_key,
            "capture_trace_rows_sha256": str(capture.rows.trace_rows_sha256),
            "capture_rows_identity_sha256": capture_rows_identity_sha256,
            "captured_producer_state_sha256": producer_state_sha256,
            "span": span.as_dict(),
            "source_layer_index": source_layer_index,
            "target_layer_index": target_layer_index,
            "role": role,
            "cached_rows_sha256": cached_rows_sha256,
            "normalized_decoder_receipt_sha256": (
                decoder_receipt["decoder_receipt_sha256"]
            ),
            "normalized_decoder_tensor_sha256": (
                decoder_receipt["decoder_tensor_sha256"]
            ),
            "residual_chart_sha256": decoder_receipt["residual_chart_sha256"],
            "upstream_parent_and_cache_authenticated": True,
            "parent_forward_calls": 0,
            "parent_trace_recaptures": 0,
            "training_steps": 0,
        }
    )


def _rows_identical(value: Any) -> bool:
    torch = _torch()
    return bool(
        int(value.shape[0]) == 1
        or torch.equal(value[1:], value[0].unsqueeze(0).expand_as(value[1:]))
    )


def _selected_cached_attention_head_rows(
    *,
    rows: Any,
    row_indices: tuple[int, ...],
    selected_lineage: tuple[tuple[int, int], ...],
    field_name: str,
    label: str,
) -> Any:
    """Assemble cached per-head rows in the span compiler's selected order."""

    torch = _torch()
    by_layer: dict[int, Any] = {}
    selected = []
    for source_layer_index, source_head_index in selected_lineage:
        if source_layer_index not in by_layer:
            try:
                value = rows.field(source_layer_index, field_name)
            except (AttributeError, KeyError) as exc:
                raise Lfm2HybridProducerFirstFoldError(
                    f"{label} requires sealed cache field {field_name!r} for source "
                    f"layer {source_layer_index}."
                ) from exc
            by_layer[source_layer_index] = _index_rows(
                value,
                row_indices,
                f"source layer {source_layer_index} {label}",
            )
        layer_rows = by_layer[source_layer_index]
        if (
            layer_rows.ndim != 3
            or source_head_index < 0
            or source_head_index >= int(layer_rows.shape[1])
        ):
            raise Lfm2HybridProducerFirstFoldError(
                f"{label} differs from the selected attention-head topology."
            )
        selected.append(layer_rows[:, source_head_index, :].unsqueeze(1))
    if not selected:
        raise Lfm2HybridProducerFirstFoldError(
            f"{label} has no selected source-head lineage."
        )
    result = torch.cat(selected, dim=1).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2HybridProducerFirstFoldError(f"{label} contains NaN or infinity.")
    return result


def _compile_span_coupled_candidate_overlays(
    *,
    parent: Any,
    child: Any,
    spans: tuple[Lfm2ModelFoldSpan, ...],
    evidence: Lfm2ProducerFirstEvidence,
    capture: Lfm2ProducerFirstCapture,
    producer_state_sha256: str,
    capture_rows_identity_sha256: str,
    config: Lfm2ModelFoldConfig,
) -> tuple[
    Mapping[int, Mapping[str, Any]],
    Mapping[int, Mapping[str, Any]],
    Mapping[str, Any],
]:
    """Compile executable multi-layer span state from sealed trace rows."""

    from .hf_lfm2_attention_group_qdeim import (
        Lfm2AttentionGroupQdeimSpanSource,
        assemble_lfm2_attention_group_qdeim_span_sources,
        compile_lfm2_attention_group_span_qdeim,
        lfm2_attention_group_qdeim_span_context_evidence_sha256,
        lfm2_attention_group_qdeim_span_source_state_sha256,
        lfm2_attention_group_qdeim_target_template_sha256,
    )
    from .hf_lfm2_canonical import Lfm2AttentionContract
    from .hf_lfm2_cpcf_attention_dimension_fold import lfm2_runtime_rms_norm
    from .hf_lfm2_mlp_qdeim import (
        Lfm2MlpQdeimSpanSource,
        assemble_lfm2_mlp_qdeim_span_sources,
        compile_lfm2_mlp_span_coupled_nonlinear_qdeim,
        lfm2_mlp_qdeim_source_state_sha256,
        lfm2_mlp_qdeim_span_functional_evidence_sha256,
        lfm2_mlp_qdeim_target_template_sha256,
    )
    from .hf_lfm2_span_shared_qk_norm_refit import (
        compile_lfm2_span_shared_qk_norm_refit,
    )

    torch = _torch()
    parent_layers = _layers(parent)
    child_layers = _layers(child)
    depth_spans = tuple(span for span in spans if span.is_depth_fold)
    overrides: dict[int, Mapping[str, Any]] = {}
    receipts: dict[int, Mapping[str, Any]] = {}
    row_identity_sha256 = _canonical_sha256(
        {
            "schema": "neural-forge-lfm2-span-aligned-row-identity-v1",
            "parent_rows_identity_sha256": evidence.parent_rows_identity_sha256,
            "selection_indices": list(evidence.selection_indices),
            "ordered_paths": [
                str(evidence.parent_rows.paths[index])
                for index in evidence.selection_indices
            ],
        }
    )
    for span in depth_spans:
        receipt_schema = _SPAN_COUPLED_PRODUCTION_RECEIPT_V1
        target_index = span.target_layer_index
        target_layer = child_layers[target_index]
        target_local_state = target_layer.state_dict()
        target_mlp_names = (
            "feed_forward.w1.weight",
            "feed_forward.w3.weight",
            "feed_forward.w2.weight",
        )
        if any(
            name not in target_local_state
            or target_local_state[name].dtype != torch.bfloat16
            for name in target_mlp_names
        ):
            body = {
                "schema": receipt_schema,
                "span": span.as_dict(),
                "parent_identity_sha256": evidence.parent_identity_sha256,
                "parent_cache_fingerprint": evidence.parent_cache_fingerprint,
                "parent_rows_identity_sha256": evidence.parent_rows_identity_sha256,
                "producer_evidence_sha256": evidence.evidence_sha256,
                "capture_trace_rows_sha256": str(capture.rows.trace_rows_sha256),
                "capture_rows_identity_sha256": capture_rows_identity_sha256,
                "captured_producer_state_sha256": producer_state_sha256,
                "installed": False,
                "installed_any": False,
                "installed_all": False,
                "reason": "target-span-state-is-not-dense-bfloat16",
                "legacy_anchor_candidate_preserved": True,
                "numeric_quality_gate_used": False,
                "training_steps": 0,
            }
            receipts[target_index] = MappingProxyType(
                {**body, "receipt_sha256": _canonical_sha256(body)}
            )
            continue

        mlp_sources_by_layer: dict[int, Lfm2MlpQdeimSpanSource] = {}
        mlp_decoder_receipts: dict[int, Mapping[str, Any]] = {}
        for source_index in span.source_layer_indices:
            source_layer = parent_layers[source_index]
            decoder, chart, decoder_receipt = _compile_normalized_span_decoder(
                parent_model=parent,
                child_model=child,
                parent_layer=source_layer,
                target_layer=target_layer,
                parent_rows=evidence.parent_rows,
                child_rows=capture.rows,
                source_layer_index=source_index,
                target_layer_index=target_index,
                fit_indices=evidence.fit_indices,
                heldout_indices=evidence.validation_indices,
                role="mlp",
            )
            cached_post_swiglu = _index_rows(
                evidence.parent_rows.field(source_index, "mlp.post_swiglu"),
                evidence.selection_indices,
                f"source layer {source_index} cached post-SwiGLU rows",
            )
            cached_sha256 = _authenticated_tensor_sha256(cached_post_swiglu)
            source_state_sha256 = lfm2_mlp_qdeim_source_state_sha256(
                source_layer.feed_forward.w1.weight,
                source_layer.feed_forward.w3.weight,
                source_layer.feed_forward.w2.weight,
            )
            cache_receipt_sha256 = _span_source_receipt_sha256(
                evidence=evidence,
                capture=capture,
                producer_state_sha256=producer_state_sha256,
                capture_rows_identity_sha256=capture_rows_identity_sha256,
                span=span,
                source_layer_index=source_index,
                target_layer_index=target_index,
                role="mlp",
                cached_rows_sha256=cached_sha256,
                decoder_receipt=decoder_receipt,
            )
            mlp_sources_by_layer[source_index] = Lfm2MlpQdeimSpanSource(
                source_layer_index=source_index,
                source_gate_weight=source_layer.feed_forward.w1.weight,
                source_up_weight=source_layer.feed_forward.w3.weight,
                source_down_weight=source_layer.feed_forward.w2.weight,
                expected_source_state_sha256=source_state_sha256,
                cached_parent_post_swiglu_rows=cached_post_swiglu,
                expected_cached_parent_post_swiglu_sha256=cached_sha256,
                cache_receipt_sha256=cache_receipt_sha256,
                normalized_input_decoder=decoder,
                expected_normalized_input_decoder_sha256=(
                    _authenticated_tensor_sha256(decoder)
                ),
                residual_output_chart=chart,
                expected_residual_output_chart_sha256=(
                    _authenticated_tensor_sha256(chart)
                ),
            )
            mlp_decoder_receipts[source_index] = decoder_receipt
        mlp_sources = assemble_lfm2_mlp_qdeim_span_sources(
            span=span,
            sources_by_layer=mlp_sources_by_layer,
        )
        mlp_evidence_sha256 = lfm2_mlp_qdeim_span_functional_evidence_sha256(
            span=span,
            sources=mlp_sources,
        )
        mlp_template_sha256 = lfm2_mlp_qdeim_target_template_sha256(
            target_local_state[target_mlp_names[0]],
            target_local_state[target_mlp_names[1]],
            target_local_state[target_mlp_names[2]],
        )
        compiled_mlp = compile_lfm2_mlp_span_coupled_nonlinear_qdeim(
            span=span,
            sources=mlp_sources,
            expected_functional_evidence_sha256=mlp_evidence_sha256,
            target_gate_weight_template=target_local_state[target_mlp_names[0]],
            target_up_weight_template=target_local_state[target_mlp_names[1]],
            target_down_weight_template=target_local_state[target_mlp_names[2]],
            expected_target_template_sha256=mlp_template_sha256,
            compute_device="cpu",
        )
        local_override = {
            f"feed_forward.{name}": value
            for name, value in compiled_mlp.state_dict().items()
        }
        mlp_report = compiled_mlp.as_dict()

        attention_report: dict[str, Any] = {
            "installed": False,
            "reason": "target-span-operator-is-not-attention",
        }
        target_contract = inspect_lfm2_layer(target_layer)
        attention_source_indices = tuple(
            source_index
            for source_index in span.source_layer_indices
            if inspect_lfm2_layer(parent_layers[source_index]).kind == "full_attention"
        )
        if (
            target_contract.kind == "full_attention"
            and isinstance(target_contract.operator, Lfm2AttentionContract)
            and attention_source_indices
        ):
            target_attention = target_layer.self_attn
            total_source_groups = sum(
                inspect_lfm2_layer(parent_layers[index]).operator.key_value_heads
                for index in attention_source_indices
            )
            if (
                target_contract.operator.key_value_heads
                >= len(attention_source_indices)
                and target_contract.operator.key_value_heads < total_source_groups
            ):
                attention_sources_by_layer: dict[
                    int, Lfm2AttentionGroupQdeimSpanSource
                ] = {}
                attention_decoder_receipts: dict[int, Mapping[str, Any]] = {}
                for source_index in attention_source_indices:
                    source_layer = parent_layers[source_index]
                    source_contract = inspect_lfm2_layer(source_layer)
                    assert isinstance(source_contract.operator, Lfm2AttentionContract)
                    decoder, chart, decoder_receipt = _compile_normalized_span_decoder(
                        parent_model=parent,
                        child_model=child,
                        parent_layer=source_layer,
                        target_layer=target_layer,
                        parent_rows=evidence.parent_rows,
                        child_rows=capture.rows,
                        source_layer_index=source_index,
                        target_layer_index=target_index,
                        fit_indices=evidence.fit_indices,
                        heldout_indices=evidence.validation_indices,
                        role="attention",
                    )
                    cached_context = _index_rows(
                        evidence.parent_rows.field(
                            source_index,
                            "attention.context",
                        ),
                        evidence.selection_indices,
                        f"source layer {source_index} cached attention context",
                    )
                    cached_sha256 = _authenticated_tensor_sha256(cached_context)
                    source_attention = source_layer.self_attn
                    source_state_sha256 = (
                        lfm2_attention_group_qdeim_span_source_state_sha256(
                            source_attention.q_proj.weight,
                            source_attention.k_proj.weight,
                            source_attention.v_proj.weight,
                            source_attention.out_proj.weight,
                            source_attention.q_layernorm.weight,
                            source_attention.k_layernorm.weight,
                        )
                    )
                    cache_receipt_sha256 = _span_source_receipt_sha256(
                        evidence=evidence,
                        capture=capture,
                        producer_state_sha256=producer_state_sha256,
                        capture_rows_identity_sha256=capture_rows_identity_sha256,
                        span=span,
                        source_layer_index=source_index,
                        target_layer_index=target_index,
                        role="attention",
                        cached_rows_sha256=cached_sha256,
                        decoder_receipt=decoder_receipt,
                    )
                    attention_sources_by_layer[source_index] = (
                        Lfm2AttentionGroupQdeimSpanSource(
                            source_layer_index=source_index,
                            source_query_weight=source_attention.q_proj.weight,
                            source_key_weight=source_attention.k_proj.weight,
                            source_value_weight=source_attention.v_proj.weight,
                            source_out_weight=source_attention.out_proj.weight,
                            expected_source_state_sha256=source_state_sha256,
                            source_query_head_count=(
                                source_contract.operator.query_heads
                            ),
                            source_key_value_head_count=(
                                source_contract.operator.key_value_heads
                            ),
                            cached_parent_post_softmax_context_rows=cached_context,
                            expected_cached_parent_post_softmax_context_sha256=(
                                cached_sha256
                            ),
                            cache_receipt_sha256=cache_receipt_sha256,
                            aligned_row_identity_sha256=row_identity_sha256,
                            normalized_input_decoder=decoder,
                            expected_normalized_input_decoder_sha256=(
                                _authenticated_tensor_sha256(decoder)
                            ),
                            residual_output_chart=chart,
                            expected_residual_output_chart_sha256=(
                                _authenticated_tensor_sha256(chart)
                            ),
                            source_query_norm_weight=(
                                source_attention.q_layernorm.weight
                            ),
                            source_key_norm_weight=(
                                source_attention.k_layernorm.weight
                            ),
                        )
                    )
                    attention_decoder_receipts[source_index] = decoder_receipt
                attention_sources = assemble_lfm2_attention_group_qdeim_span_sources(
                    span=span,
                    attention_source_layer_indices=attention_source_indices,
                    sources_by_layer=attention_sources_by_layer,
                )
                context_evidence_sha256 = (
                    lfm2_attention_group_qdeim_span_context_evidence_sha256(
                        span=span,
                        sources=attention_sources,
                    )
                )
                attention_template_sha256 = (
                    lfm2_attention_group_qdeim_target_template_sha256(
                        target_attention.q_proj.weight,
                        target_attention.k_proj.weight,
                        target_attention.v_proj.weight,
                        target_attention.out_proj.weight,
                    )
                )
                compiled_attention = compile_lfm2_attention_group_span_qdeim(
                    span=span,
                    sources=attention_sources,
                    expected_context_evidence_sha256=context_evidence_sha256,
                    target_query_weight_template=target_attention.q_proj.weight,
                    target_key_weight_template=target_attention.k_proj.weight,
                    target_value_weight_template=target_attention.v_proj.weight,
                    target_out_weight_template=target_attention.out_proj.weight,
                    expected_target_template_sha256=attention_template_sha256,
                    target_query_head_count=(target_contract.operator.query_heads),
                    target_key_value_head_count=(
                        target_contract.operator.key_value_heads
                    ),
                    compute_device="cpu",
                )
                q_norm_rows = compiled_attention.selected_query_norm_weight_rows
                k_norm_rows = compiled_attention.selected_key_norm_weight_rows
                shared_norm_representable = bool(
                    q_norm_rows is not None
                    and k_norm_rows is not None
                    and _rows_identical(q_norm_rows)
                    and _rows_identical(k_norm_rows)
                )
                raw_attention_report = compiled_attention.as_dict()
                if shared_norm_representable:
                    local_override.update(
                        {
                            f"self_attn.{name}": value
                            for name, value in compiled_attention.state_dict().items()
                        }
                    )
                    local_override.update(
                        {
                            "self_attn.q_layernorm.weight": q_norm_rows[0]
                            .detach()
                            .cpu()
                            .contiguous()
                            .clone(),
                            "self_attn.k_layernorm.weight": k_norm_rows[0]
                            .detach()
                            .cpu()
                            .contiguous()
                            .clone(),
                        }
                    )
                    attention_report = {
                        "installed": True,
                        "shared_qk_norm_directly_representable": True,
                        "heterogeneous_norm_rows_averaged": False,
                        "compiler": raw_attention_report,
                        "decoder_receipt_sha256_by_source": {
                            str(index): value["decoder_receipt_sha256"]
                            for index, value in attention_decoder_receipts.items()
                        },
                    }
                else:
                    child_selection_inputs = lfm2_runtime_rms_norm(
                        _index_rows(
                            capture.rows.field(target_index, "residual.pre"),
                            evidence.selection_indices,
                            (f"target layer {target_index} attention selection input"),
                        ),
                        _layer_norm_weight(target_layer, "operator_norm"),
                        _norm_epsilon(child, "child"),
                    )
                    query_selection_targets = _selected_cached_attention_head_rows(
                        rows=evidence.parent_rows,
                        row_indices=evidence.selection_indices,
                        selected_lineage=(
                            compiled_attention.selected_source_query_heads
                        ),
                        field_name="attention.q_normalized_pre_rope",
                        label="cached normalized pre-RoPE query rows",
                    )
                    key_selection_targets = _selected_cached_attention_head_rows(
                        rows=evidence.parent_rows,
                        row_indices=evidence.selection_indices,
                        selected_lineage=(
                            compiled_attention.selected_source_key_value_groups
                        ),
                        field_name="attention.k_normalized_pre_rope",
                        label="cached normalized pre-RoPE key rows",
                    )
                    diagnostic_inputs = None
                    diagnostic_query_targets = None
                    diagnostic_key_targets = None
                    if evidence.validation_indices:
                        diagnostic_inputs = lfm2_runtime_rms_norm(
                            _index_rows(
                                capture.rows.field(target_index, "residual.pre"),
                                evidence.validation_indices,
                                (
                                    f"target layer {target_index} attention diagnostic "
                                    "input"
                                ),
                            ),
                            _layer_norm_weight(target_layer, "operator_norm"),
                            _norm_epsilon(child, "child"),
                        )
                        diagnostic_query_targets = _selected_cached_attention_head_rows(
                            rows=evidence.parent_rows,
                            row_indices=evidence.validation_indices,
                            selected_lineage=(
                                compiled_attention.selected_source_query_heads
                            ),
                            field_name="attention.q_normalized_pre_rope",
                            label=("diagnostic normalized pre-RoPE query rows"),
                        )
                        diagnostic_key_targets = _selected_cached_attention_head_rows(
                            rows=evidence.parent_rows,
                            row_indices=evidence.validation_indices,
                            selected_lineage=(
                                compiled_attention.selected_source_key_value_groups
                            ),
                            field_name="attention.k_normalized_pre_rope",
                            label="diagnostic normalized pre-RoPE key rows",
                        )
                    compiled_attention_state = compiled_attention.state_dict()
                    norm_refit = compile_lfm2_span_shared_qk_norm_refit(
                        transported_query_weight=(
                            compiled_attention_state["q_proj.weight"]
                        ),
                        transported_key_weight=(
                            compiled_attention_state["k_proj.weight"]
                        ),
                        unchanged_value_weight=(
                            compiled_attention_state["v_proj.weight"]
                        ),
                        unchanged_out_weight=(
                            compiled_attention_state["out_proj.weight"]
                        ),
                        selected_query_source_norm_rows=q_norm_rows,
                        selected_key_source_norm_rows=k_norm_rows,
                        cached_child_normalized_input_rows=(child_selection_inputs),
                        cached_query_normalized_pre_rope_rows=(query_selection_targets),
                        cached_key_normalized_pre_rope_rows=(key_selection_targets),
                        target_query_norm_weight_template=(
                            target_attention.q_layernorm.weight
                        ),
                        target_key_norm_weight_template=(
                            target_attention.k_layernorm.weight
                        ),
                        child_norm_epsilon=_norm_epsilon(child, "child"),
                        ridge=config.layer_config.endpoint_ridge,
                        selected_source_query_heads=(
                            compiled_attention.selected_source_query_heads
                        ),
                        selected_source_key_value_groups=(
                            compiled_attention.selected_source_key_value_groups
                        ),
                        parent_attention_compiler_result_sha256=(
                            raw_attention_report["result_sha256"]
                        ),
                        diagnostic_child_normalized_input_rows=diagnostic_inputs,
                        diagnostic_query_normalized_pre_rope_rows=(
                            diagnostic_query_targets
                        ),
                        diagnostic_key_normalized_pre_rope_rows=(
                            diagnostic_key_targets
                        ),
                    )
                    local_override.update(
                        {
                            f"self_attn.{name}": value
                            for name, value in compiled_attention_state.items()
                        }
                    )
                    local_override.update(
                        {
                            f"self_attn.{name}": value
                            for name, value in norm_refit.state_dict().items()
                        }
                    )
                    receipt_schema = _SPAN_COUPLED_PRODUCTION_RECEIPT_V2
                    attention_report = {
                        "installed": True,
                        "shared_qk_norm_directly_representable": False,
                        "heterogeneous_norm_rows_averaged": False,
                        "shared_qk_norm_refit": norm_refit.as_dict(),
                        "legacy_anchor_attention_preserved": False,
                        "compiler": raw_attention_report,
                        "decoder_receipt_sha256_by_source": {
                            str(index): value["decoder_receipt_sha256"]
                            for index, value in attention_decoder_receipts.items()
                        },
                    }
            else:
                attention_report = {
                    "installed": False,
                    "reason": "span-attention-union-does-not-reduce-into-target-groups",
                    "legacy_anchor_attention_preserved": True,
                }

        installed_state_sha256 = {
            name: fingerprint_lfm2_model_fold_tensor(value)
            for name, value in sorted(local_override.items())
        }
        body = {
            "schema": receipt_schema,
            "span": span.as_dict(),
            "parent_identity_sha256": evidence.parent_identity_sha256,
            "parent_cache_fingerprint": evidence.parent_cache_fingerprint,
            "parent_rows_identity_sha256": evidence.parent_rows_identity_sha256,
            "producer_evidence_sha256": evidence.evidence_sha256,
            "capture_trace_rows_sha256": str(capture.rows.trace_rows_sha256),
            "capture_rows_identity_sha256": capture_rows_identity_sha256,
            "captured_producer_state_sha256": producer_state_sha256,
            "mlp": {
                "installed": True,
                "compiler": mlp_report,
                "decoder_receipt_sha256_by_source": {
                    str(index): value["decoder_receipt_sha256"]
                    for index, value in mlp_decoder_receipts.items()
                },
            },
            "attention": attention_report,
            "installed_local_state_sha256": installed_state_sha256,
            "installed_local_state_hash_scheme": (MODEL_FOLD_TENSOR_HASH_SCHEME),
            "installed": True,
            "installed_any": True,
            "installed_all": bool(
                target_contract.kind != "full_attention"
                or attention_report.get("installed") is True
            ),
            "dense_ordinary_target_tensors": True,
            "all_mlp_span_sources_consumed": True,
            "numeric_quality_gate_used": False,
            "manual_binary_gate_used": False,
            "parent_forward_calls": 0,
            "parent_trace_recaptures": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
        }
        receipts[target_index] = MappingProxyType(
            {**body, "receipt_sha256": _canonical_sha256(body)}
        )
        overrides[target_index] = MappingProxyType(local_override)

    root_body = {
        "schema": "neural-forge-lfm2-span-coupled-production-root-v1",
        "parent_identity_sha256": evidence.parent_identity_sha256,
        "parent_cache_fingerprint": evidence.parent_cache_fingerprint,
        "parent_rows_identity_sha256": evidence.parent_rows_identity_sha256,
        "producer_evidence_sha256": evidence.evidence_sha256,
        "captured_producer_state_sha256": producer_state_sha256,
        "capture_cache_key": capture.cache_key,
        "capture_trace_rows_sha256": str(capture.rows.trace_rows_sha256),
        "capture_rows_identity_sha256": capture_rows_identity_sha256,
        "depth_spans": [span.as_dict() for span in depth_spans],
        "span_receipt_sha256": {
            str(index): value["receipt_sha256"]
            for index, value in sorted(receipts.items())
        },
    }
    root_receipt = MappingProxyType(
        {
            **root_body,
            "provenance_root_sha256": _canonical_sha256(root_body),
        }
    )
    return (
        MappingProxyType(overrides),
        MappingProxyType(receipts),
        root_receipt,
    )


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
            "producer_anchor_copied_tensors": (self.producer_anchor_copied_tensors),
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
            "seed_erasure_by_construction": (self.seed_erasure_by_construction),
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
        if capture.cache_key != getattr(capture.rows, "cache_key", None):
            raise Lfm2HybridProducerFirstFoldError(
                "Captured child rows are not bound to their declared cache key."
            )
        capture_rows_identity_sha256 = _rows_identity_sha256(capture.rows)
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
        model_evidence = assemble_lfm2_model_fold_evidence_from_trace_rows(
            cache_fingerprint=evidence.parent_cache_fingerprint,
            child_feature_state_sha256=producer_state,
            spans=span_values,
            parent_rows=evidence.parent_rows,
            child_rows=capture.rows,
            selection_indices=evidence.selection_indices,
            fit_indices=evidence.fit_indices,
            validation_indices=evidence.validation_indices,
        )
        model_plan = build_lfm2_model_fold_plan(
            parent,
            child,
            span_values,
            model_evidence,
            config=selected,
        )
        if any(span.is_depth_fold for span in span_values):
            (
                span_overrides,
                span_receipts,
                span_provenance_receipt,
            ) = _compile_span_coupled_candidate_overlays(
                parent=parent,
                child=child,
                spans=span_values,
                evidence=evidence,
                capture=capture,
                producer_state_sha256=producer_state,
                capture_rows_identity_sha256=capture_rows_identity_sha256,
                config=selected,
            )
            model_plan = overlay_lfm2_span_coupled_candidates(
                child,
                model_plan,
                local_state_overrides=span_overrides,
                span_receipts=span_receipts,
                provenance_receipt=span_provenance_receipt,
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
        if _rows_identity_sha256(capture.rows) != capture_rows_identity_sha256:
            raise Lfm2HybridProducerFirstFoldError(
                "Captured child trace rows changed during span compilation."
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
