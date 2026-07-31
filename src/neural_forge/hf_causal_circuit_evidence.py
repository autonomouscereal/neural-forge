"""Immutable one-forward evidence for dense Hugging Face causal folding.

The compiler consumes this object instead of a live parent.  One call to
``capture_hf_causal_circuit_evidence`` executes the parent exactly once via
the complete transport tracer, copies every required tensor to CPU, and
seals the resulting bundle.  Fit and validation rows are disjoint views of
that one capture; collecting a second parent trace is neither required nor
exposed by this module.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .dynamic_circuit_ir import canonical_sha256
from .hf_arch_layout import DecoderLayout, build_decoder_layout, resolve_module
from .hf_transport_trace import collect_hf_transport_trace


SCHEMA = "neural-forge-hf-causal-circuit-evidence-v1"


class HFCausalCircuitEvidenceError(ValueError):
    """The cached parent evidence is incomplete, mutable, or incompatible."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise HFCausalCircuitEvidenceError(
            "PyTorch is required for HF causal circuit evidence."
        ) from exc
    return torch


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HFCausalCircuitEvidenceError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HFCausalCircuitEvidenceError(f"{name} must be a positive integer.")
    return value


def _finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    import math

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HFCausalCircuitEvidenceError(f"{name} must be finite.") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and " if positive else ""
        raise HFCausalCircuitEvidenceError(
            f"{name} must be {qualifier}finite."
        )
    return result


def _clone_tensor(value: Any, name: str, *, ndim: int | None = None) -> Any:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise HFCausalCircuitEvidenceError(f"{name} must be a tensor.")
    if ndim is not None and value.ndim != ndim:
        raise HFCausalCircuitEvidenceError(
            f"{name} must be rank {ndim}, got rank {value.ndim}."
        )
    result = value.detach().to(device="cpu").contiguous().clone()
    if result.requires_grad or result.grad_fn is not None:
        raise HFCausalCircuitEvidenceError(f"{name} retained a gradient graph.")
    if (result.is_floating_point() or result.is_complex()) and not bool(
        result.isfinite().all().item()
    ):
        raise HFCausalCircuitEvidenceError(f"{name} contains non-finite values.")
    return result


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    tensor = _clone_tensor(value, "sealed tensor")
    raw = tensor.view(torch.uint8).numpy().tobytes()
    return canonical_sha256(
        {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": [int(size) for size in tensor.shape],
            "payload_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )


def _canonical_json(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    try:
        decoded = json.loads(
            json.dumps(
                dict(value),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as exc:
        raise HFCausalCircuitEvidenceError(
            f"{name} must be canonical JSON."
        ) from exc
    return MappingProxyType(decoded)


@dataclass(frozen=True, slots=True)
class HFCausalArchitectureEvidence:
    family: str
    hidden_size: int
    intermediate_size: int
    layer_count: int
    vocabulary_size: int
    query_heads: int
    key_value_heads: int
    head_dim: int
    rms_norm_epsilon: float
    rope_theta: float
    rope_type: str
    max_position_embeddings: int
    attention_scaling: float
    attention_projection_bias: bool
    mlp_projection_bias: bool
    query_key_norm: str
    tie_word_embeddings: bool

    def __post_init__(self) -> None:
        family = str(self.family).strip().lower()
        if family not in {"llama", "qwen3"}:
            raise HFCausalCircuitEvidenceError(
                "Full-circuit evidence currently supports Llama and Qwen3 layouts."
            )
        object.__setattr__(self, "family", family)
        for name in (
            "hidden_size",
            "intermediate_size",
            "layer_count",
            "vocabulary_size",
            "query_heads",
            "key_value_heads",
            "head_dim",
            "max_position_embeddings",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if self.query_heads % self.key_value_heads:
            raise HFCausalCircuitEvidenceError(
                "query_heads must be divisible by key_value_heads."
            )
        if self.head_dim % 2:
            raise HFCausalCircuitEvidenceError("RoPE head_dim must be even.")
        object.__setattr__(
            self,
            "rms_norm_epsilon",
            _finite_float(self.rms_norm_epsilon, "rms_norm_epsilon", positive=True),
        )
        object.__setattr__(
            self,
            "rope_theta",
            _finite_float(self.rope_theta, "rope_theta", positive=True),
        )
        rope_type = str(self.rope_type).strip().lower()
        if not rope_type:
            raise HFCausalCircuitEvidenceError("rope_type must be nonempty.")
        object.__setattr__(self, "rope_type", rope_type)
        object.__setattr__(
            self,
            "attention_scaling",
            _finite_float(self.attention_scaling, "attention_scaling", positive=True),
        )
        for name in (
            "attention_projection_bias",
            "mlp_projection_bias",
            "tie_word_embeddings",
        ):
            if not isinstance(getattr(self, name), bool):
                raise HFCausalCircuitEvidenceError(f"{name} must be boolean.")
        norm = str(self.query_key_norm).strip().lower()
        if norm not in {"identity", "rms_norm"}:
            raise HFCausalCircuitEvidenceError(
                "query_key_norm must be 'identity' or 'rms_norm'."
            )
        object.__setattr__(self, "query_key_norm", norm)

    def as_dict(self) -> dict[str, Any]:
        return {
            descriptor.name: getattr(self, descriptor.name)
            for descriptor in fields(self)
        }


@dataclass(frozen=True, slots=True)
class HFCausalLayerEvidence:
    """Complete cached circuit states for one source decoder layer."""

    index: int
    block_input: Any
    attention_norm_output: Any
    query_projection_output: Any
    key_projection_output: Any
    value_projection_output: Any
    attention_context: Any
    attention_delta: Any
    post_attention_residual: Any
    mlp_norm_output: Any
    mlp_gate_output: Any
    mlp_up_output: Any
    mlp_delta: Any
    block_output: Any
    query_norm_weight: Any | None = None
    key_norm_weight: Any | None = None

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise HFCausalCircuitEvidenceError(
                "layer evidence index must be a non-negative integer."
            )
        for descriptor in fields(self):
            if descriptor.name in {"index", "query_norm_weight", "key_norm_weight"}:
                continue
            object.__setattr__(
                self,
                descriptor.name,
                _clone_tensor(getattr(self, descriptor.name), descriptor.name, ndim=3),
            )
        for name in ("query_norm_weight", "key_norm_weight"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _clone_tensor(value, name, ndim=1))

    def tensor_items(self) -> tuple[tuple[str, Any], ...]:
        return tuple(
            (descriptor.name, getattr(self, descriptor.name))
            for descriptor in fields(self)
            if descriptor.name != "index" and getattr(self, descriptor.name) is not None
        )


class HFCausalCircuitEvidence:
    """A self-verifying immutable parent circuit cache."""

    __slots__ = (
        "parent_graph_fingerprint",
        "architecture",
        "input_ids",
        "attention_mask",
        "position_ids",
        "valid_token_mask",
        "fit_token_mask",
        "validation_token_mask",
        "token_embedding_weight",
        "lm_head_weight",
        "layers",
        "final_norm_output",
        "logits",
        "metadata",
        "parent_forward_calls",
        "_tensor_manifest",
        "_cache_key",
    )

    def __init__(
        self,
        *,
        parent_graph_fingerprint: str,
        architecture: HFCausalArchitectureEvidence,
        input_ids: Any,
        attention_mask: Any,
        position_ids: Any,
        valid_token_mask: Any,
        fit_token_mask: Any,
        validation_token_mask: Any,
        token_embedding_weight: Any,
        lm_head_weight: Any,
        layers: tuple[HFCausalLayerEvidence, ...],
        final_norm_output: Any,
        logits: Any,
        metadata: Mapping[str, Any] | None = None,
        parent_forward_calls: int = 1,
    ) -> None:
        self.parent_graph_fingerprint = _digest(
            parent_graph_fingerprint, "parent_graph_fingerprint"
        )
        if not isinstance(architecture, HFCausalArchitectureEvidence):
            raise HFCausalCircuitEvidenceError(
                "architecture must be HFCausalArchitectureEvidence."
            )
        self.architecture = architecture
        self.input_ids = _clone_tensor(input_ids, "input_ids", ndim=2)
        self.attention_mask = _clone_tensor(attention_mask, "attention_mask", ndim=2)
        self.position_ids = _clone_tensor(position_ids, "position_ids", ndim=2)
        self.valid_token_mask = _clone_tensor(
            valid_token_mask, "valid_token_mask", ndim=2
        ).to(dtype=_torch().bool)
        self.fit_token_mask = _clone_tensor(
            fit_token_mask, "fit_token_mask", ndim=2
        ).to(dtype=_torch().bool)
        self.validation_token_mask = _clone_tensor(
            validation_token_mask, "validation_token_mask", ndim=2
        ).to(dtype=_torch().bool)
        shape = tuple(int(size) for size in self.input_ids.shape)
        for name in (
            "attention_mask",
            "position_ids",
            "valid_token_mask",
            "fit_token_mask",
            "validation_token_mask",
        ):
            if tuple(int(size) for size in getattr(self, name).shape) != shape:
                raise HFCausalCircuitEvidenceError(
                    f"{name} does not match input_ids batch/sequence shape."
                )
        if self.input_ids.dtype == _torch().bool or self.input_ids.is_floating_point():
            raise HFCausalCircuitEvidenceError("input_ids must contain integers.")
        if bool((self.fit_token_mask & self.validation_token_mask).any().item()):
            raise HFCausalCircuitEvidenceError("Fit and validation masks overlap.")
        if not bool(self.fit_token_mask.any().item()) or not bool(
            self.validation_token_mask.any().item()
        ):
            raise HFCausalCircuitEvidenceError(
                "Fit and validation masks must both select cached tokens."
            )
        if bool(
            ((self.fit_token_mask | self.validation_token_mask) & ~self.valid_token_mask)
            .any()
            .item()
        ):
            raise HFCausalCircuitEvidenceError(
                "Fit/validation masks cannot select invalid tokens."
            )
        self.token_embedding_weight = _clone_tensor(
            token_embedding_weight, "token_embedding_weight", ndim=2
        )
        self.lm_head_weight = _clone_tensor(lm_head_weight, "lm_head_weight", ndim=2)
        if tuple(self.token_embedding_weight.shape) != (
            architecture.vocabulary_size,
            architecture.hidden_size,
        ) or tuple(self.lm_head_weight.shape) != (
            architecture.vocabulary_size,
            architecture.hidden_size,
        ):
            raise HFCausalCircuitEvidenceError(
                "Cached token-interface weights disagree with the architecture."
            )
        self.layers = tuple(layers)
        if (
            len(self.layers) != architecture.layer_count
            or any(not isinstance(value, HFCausalLayerEvidence) for value in self.layers)
            or tuple(value.index for value in self.layers)
            != tuple(range(architecture.layer_count))
        ):
            raise HFCausalCircuitEvidenceError(
                "Layer evidence must cover every source layer exactly once in order."
            )
        self.final_norm_output = _clone_tensor(
            final_norm_output, "final_norm_output", ndim=3
        )
        self.logits = _clone_tensor(logits, "logits", ndim=3)
        expected_residual = (*shape, architecture.hidden_size)
        expected_logits = (*shape, architecture.vocabulary_size)
        if tuple(self.final_norm_output.shape) != expected_residual:
            raise HFCausalCircuitEvidenceError(
                "final_norm_output has the wrong residual shape."
            )
        if tuple(self.logits.shape) != expected_logits:
            raise HFCausalCircuitEvidenceError("logits has the wrong vocabulary shape.")
        for layer in self.layers:
            if tuple(layer.block_input.shape) != expected_residual or tuple(
                layer.block_output.shape
            ) != expected_residual:
                raise HFCausalCircuitEvidenceError(
                    f"Layer {layer.index} residual evidence has the wrong shape."
                )
        self.metadata = _canonical_json(metadata or {}, "evidence metadata")
        if parent_forward_calls != 1:
            raise HFCausalCircuitEvidenceError(
                "A full-circuit evidence bundle must come from exactly one parent forward."
            )
        self.parent_forward_calls = 1
        self._tensor_manifest = MappingProxyType(
            {name: _tensor_sha256(value) for name, value in self.tensor_items()}
        )
        self._cache_key = canonical_sha256(self._body())
        self.verify()

    @property
    def cache_key(self) -> str:
        return self._cache_key

    @property
    def fingerprint(self) -> str:
        return self._cache_key

    @property
    def tensor_manifest(self) -> Mapping[str, str]:
        return self._tensor_manifest

    def tensor_items(self) -> tuple[tuple[str, Any], ...]:
        values: list[tuple[str, Any]] = [
            ("input_ids", self.input_ids),
            ("attention_mask", self.attention_mask),
            ("position_ids", self.position_ids),
            ("valid_token_mask", self.valid_token_mask),
            ("fit_token_mask", self.fit_token_mask),
            ("validation_token_mask", self.validation_token_mask),
            ("token_embedding_weight", self.token_embedding_weight),
            ("lm_head_weight", self.lm_head_weight),
            ("final_norm_output", self.final_norm_output),
            ("logits", self.logits),
        ]
        for layer in self.layers:
            values.extend(
                (f"layers.{layer.index}.{name}", value)
                for name, value in layer.tensor_items()
            )
        return tuple(values)

    def _body(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "parent_graph_fingerprint": self.parent_graph_fingerprint,
            "architecture": self.architecture.as_dict(),
            "tensor_manifest": dict(self._tensor_manifest),
            "metadata": dict(self.metadata),
            "parent_forward_calls": self.parent_forward_calls,
            "fit_validation_disjoint": True,
            "cached_parent_evidence_only": True,
        }

    def verify(self) -> "HFCausalCircuitEvidence":
        observed = {name: _tensor_sha256(value) for name, value in self.tensor_items()}
        if observed != dict(self._tensor_manifest) or canonical_sha256(
            self._body()
        ) != self._cache_key:
            raise HFCausalCircuitEvidenceError(
                "HF causal evidence tensors or metadata changed after sealing."
            )
        return self

    def save(self, root: str | Path) -> Path:
        """Persist a content-addressed bundle without executing any model."""

        torch = _torch()
        destination = Path(root)
        destination.mkdir(parents=True, exist_ok=True)
        payload = {
            "architecture": self.architecture.as_dict(),
            "parent_graph_fingerprint": self.parent_graph_fingerprint,
            "metadata": dict(self.metadata),
            "parent_forward_calls": self.parent_forward_calls,
            "tensors": {name: value.clone() for name, value in self.tensor_items()},
        }
        torch.save(payload, destination / "tensors.pt")
        manifest = {**self._body(), "cache_key": self.cache_key}
        (destination / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
        )
        return destination

    @classmethod
    def open(cls, root: str | Path) -> "HFCausalCircuitEvidence":
        torch = _torch()
        source = Path(root)
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        payload = torch.load(source / "tensors.pt", map_location="cpu", weights_only=True)
        tensors = dict(payload["tensors"])
        architecture = HFCausalArchitectureEvidence(**payload["architecture"])
        layers = []
        for index in range(architecture.layer_count):
            prefix = f"layers.{index}."
            layer_values = {
                name.removeprefix(prefix): value
                for name, value in tensors.items()
                if name.startswith(prefix)
            }
            layers.append(HFCausalLayerEvidence(index=index, **layer_values))
        result = cls(
            parent_graph_fingerprint=payload["parent_graph_fingerprint"],
            architecture=architecture,
            input_ids=tensors["input_ids"],
            attention_mask=tensors["attention_mask"],
            position_ids=tensors["position_ids"],
            valid_token_mask=tensors["valid_token_mask"],
            fit_token_mask=tensors["fit_token_mask"],
            validation_token_mask=tensors["validation_token_mask"],
            token_embedding_weight=tensors["token_embedding_weight"],
            lm_head_weight=tensors["lm_head_weight"],
            layers=tuple(layers),
            final_norm_output=tensors["final_norm_output"],
            logits=tensors["logits"],
            metadata=payload["metadata"],
            parent_forward_calls=payload["parent_forward_calls"],
        )
        if result.cache_key != manifest.get("cache_key"):
            raise HFCausalCircuitEvidenceError(
                "Persisted evidence payload differs from its manifest seal."
            )
        return result.verify()


def _architecture_from_layout(layout: DecoderLayout) -> HFCausalArchitectureEvidence:
    if not layout.blocks:
        raise HFCausalCircuitEvidenceError("A causal evidence layout needs layers.")
    first = layout.blocks[0]
    qk_norm = (
        "rms_norm"
        if first.attention.query_norm is not None or first.attention.key_norm is not None
        else "identity"
    )
    if any(
        block.attention.query_heads != first.attention.query_heads
        or block.attention.key_value_heads != first.attention.key_value_heads
        or block.attention.head_dim != first.attention.head_dim
        or block.attention.projection_bias != first.attention.projection_bias
        or block.mlp.projection_bias != first.mlp.projection_bias
        or (block.attention.query_norm is None) != (first.attention.query_norm is None)
        or (block.attention.key_norm is None) != (first.attention.key_norm is None)
        for block in layout.blocks
    ):
        raise HFCausalCircuitEvidenceError(
            "The first generic evidence schema requires uniform dense decoder layers."
        )
    if layout.position.kind != "rope" or layout.position.rope_theta is None:
        raise HFCausalCircuitEvidenceError("Generic causal evidence requires RoPE.")
    scaling = first.attention.scaling
    if scaling is None:
        scaling = first.attention.head_dim ** -0.5
    return HFCausalArchitectureEvidence(
        family=layout.family,
        hidden_size=layout.hidden_size,
        intermediate_size=layout.intermediate_size,
        layer_count=layout.layer_count,
        vocabulary_size=layout.embeddings.vocab_size,
        query_heads=first.attention.query_heads,
        key_value_heads=first.attention.key_value_heads,
        head_dim=first.attention.head_dim,
        rms_norm_epsilon=first.attention_input_norm.epsilon,
        rope_theta=layout.position.rope_theta,
        rope_type=layout.position.rope_type or "default",
        max_position_embeddings=layout.position.max_positions,
        attention_scaling=scaling,
        attention_projection_bias=first.attention.projection_bias,
        mlp_projection_bias=first.mlp.projection_bias,
        query_key_norm=qk_norm,
        tie_word_embeddings=layout.tie_word_embeddings,
    )


def _default_split(valid: Any) -> tuple[Any, Any]:
    torch = _torch()
    indices = torch.nonzero(valid.reshape(-1), as_tuple=False).reshape(-1)
    if int(indices.numel()) < 2:
        raise HFCausalCircuitEvidenceError(
            "One-forward evidence needs at least two valid tokens for disjoint splits."
        )
    fit = torch.zeros(int(valid.numel()), dtype=torch.bool)
    validation = torch.zeros_like(fit)
    fit[indices[0::2]] = True
    validation[indices[1::2]] = True
    return fit.reshape_as(valid), validation.reshape_as(valid)


def capture_hf_causal_circuit_evidence(
    parent_model: Any,
    model_inputs: Mapping[str, Any] | Any,
    *,
    parent_graph_fingerprint: str,
    valid_token_mask: Any | None = None,
    fit_token_mask: Any | None = None,
    validation_token_mask: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> HFCausalCircuitEvidence:
    """Capture and seal all compiler evidence with exactly one parent forward."""

    torch = _torch()
    layout = build_decoder_layout(parent_model)
    architecture = _architecture_from_layout(layout)
    if isinstance(model_inputs, Mapping):
        inputs = dict(model_inputs)
    else:
        inputs = {"input_ids": model_inputs}
    if "input_ids" not in inputs or inputs.get("inputs_embeds") is not None:
        raise HFCausalCircuitEvidenceError(
            "Full-model folding evidence requires input_ids, not inputs_embeds."
        )
    input_ids = torch.as_tensor(inputs["input_ids"])
    if input_ids.ndim != 2:
        raise HFCausalCircuitEvidenceError("input_ids must be rank 2.")
    batch, sequence = (int(input_ids.shape[0]), int(input_ids.shape[1]))
    attention_mask = torch.as_tensor(
        inputs.get("attention_mask", torch.ones_like(input_ids)),
        device=input_ids.device,
    )
    if tuple(attention_mask.shape) != (batch, sequence):
        raise HFCausalCircuitEvidenceError("attention_mask has the wrong shape.")
    position_ids = torch.as_tensor(
        inputs.get(
            "position_ids",
            torch.arange(sequence, device=input_ids.device)
            .reshape(1, sequence)
            .expand(batch, sequence),
        ),
        device=input_ids.device,
    )
    if tuple(position_ids.shape) != (batch, sequence):
        raise HFCausalCircuitEvidenceError("position_ids has the wrong shape.")
    inputs["input_ids"] = input_ids
    inputs["attention_mask"] = attention_mask
    inputs["position_ids"] = position_ids
    valid = (
        attention_mask.to(dtype=torch.bool)
        if valid_token_mask is None
        else torch.as_tensor(valid_token_mask, device=input_ids.device).to(torch.bool)
    )
    if tuple(valid.shape) != (batch, sequence):
        raise HFCausalCircuitEvidenceError("valid_token_mask has the wrong shape.")
    if fit_token_mask is None and validation_token_mask is None:
        fit, validation = _default_split(valid.detach().cpu())
    elif fit_token_mask is None or validation_token_mask is None:
        raise HFCausalCircuitEvidenceError(
            "fit_token_mask and validation_token_mask must be supplied together."
        )
    else:
        fit = torch.as_tensor(fit_token_mask).to(torch.bool).cpu()
        validation = torch.as_tensor(validation_token_mask).to(torch.bool).cpu()
    # This is the only parent execution in the complete evidence path.
    trace = collect_hf_transport_trace(
        parent_model,
        inputs,
        valid_token_mask=valid,
    )
    layers = []
    for block_layout, block in zip(layout.blocks, trace.blocks, strict=True):
        query_norm_weight = None
        key_norm_weight = None
        if block_layout.attention.query_norm is not None:
            query_norm_weight = resolve_module(
                parent_model,
                block_layout.attention.query_norm.path,
                family=layout.family,
            ).weight
        if block_layout.attention.key_norm is not None:
            key_norm_weight = resolve_module(
                parent_model,
                block_layout.attention.key_norm.path,
                family=layout.family,
            ).weight
        if block.mlp_gate_projection_output is None or block.mlp_up_projection_output is None:
            raise HFCausalCircuitEvidenceError(
                "The generic evidence path requires a gated MLP in every layer."
            )
        layers.append(
            HFCausalLayerEvidence(
                index=block.index,
                block_input=block.block_input,
                attention_norm_output=block.pre_attention_norm_output,
                query_projection_output=block.query_projection_output,
                key_projection_output=block.key_projection_output,
                value_projection_output=block.value_projection_output,
                attention_context=block.attention_output_projection_input,
                attention_delta=block.attention_delta,
                post_attention_residual=block.post_attention_residual,
                mlp_norm_output=block.pre_mlp_norm_output,
                mlp_gate_output=block.mlp_gate_projection_output,
                mlp_up_output=block.mlp_up_projection_output,
                mlp_delta=block.mlp_delta,
                block_output=block.block_output,
                query_norm_weight=query_norm_weight,
                key_norm_weight=key_norm_weight,
            )
        )
    embedding = resolve_module(parent_model, layout.embeddings.token_path, family=layout.family)
    head = resolve_module(parent_model, layout.lm_head_path, family=layout.family)
    return HFCausalCircuitEvidence(
        parent_graph_fingerprint=parent_graph_fingerprint,
        architecture=architecture,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        valid_token_mask=trace.valid_token_mask,
        fit_token_mask=fit,
        validation_token_mask=validation,
        token_embedding_weight=embedding.weight,
        lm_head_weight=head.weight,
        layers=tuple(layers),
        final_norm_output=trace.final_norm_output,
        logits=trace.logits,
        metadata={
            **dict(metadata or {}),
            "trace_family": trace.family,
            "trace_valid_token_count": trace.valid_token_count,
            "trace_hook_count": len(trace.hook_counts),
            "one_parent_forward_capture": True,
        },
        parent_forward_calls=1,
    )


__all__ = [
    "SCHEMA",
    "HFCausalArchitectureEvidence",
    "HFCausalCircuitEvidence",
    "HFCausalCircuitEvidenceError",
    "HFCausalLayerEvidence",
    "capture_hf_causal_circuit_evidence",
]
