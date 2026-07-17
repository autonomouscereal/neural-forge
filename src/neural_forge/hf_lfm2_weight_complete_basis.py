"""Prompt-independent full-weight residual basis adapter for Transformers LFM2.

This module is deliberately an adapter, not another folding algorithm.  It
proves the concrete LFM2 topology with :func:`inspect_lfm2_layer`, inventories
every learned parameter alias group, and translates every parameter with a
model-residual axis into one generic
:class:`~neural_forge.dynamic_weight_complete_basis.ResidualFacingOperator`.

The only learned LFM2 parameters that do not enter the residual-width Gram are
the attention ``q_layernorm`` and ``k_layernorm`` vectors.  Their feature axis
is ``head_dim`` rather than ``hidden_size``; the coverage manifest classifies
them explicitly as head-local instead of pretending that they are residual
coordinates.  All other learned parameter groups must be represented exactly
once or inventory construction fails closed.

No model execution, trace capture, autograd, optimizer, training, or filesystem
operation is performed here.  Optional activation information is accepted only
as an already materialized bounded Gram supplement by the generic builder.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import torch

from neural_forge.dynamic_weight_complete_basis import (
    FeatureOrientation,
    ResidualFacingOperator,
    WeightCompleteBasisResult,
    build_weight_complete_basis,
)
from neural_forge.hf_lfm2_canonical import (
    Lfm2AttentionContract,
    Lfm2ShortConvContract,
    inspect_lfm2_layer,
)


LFM2_WEIGHT_INVENTORY_SCHEMA = (
    "neural-forge-lfm2-weight-complete-inventory-v1"
)
LFM2_WEIGHT_BASIS_SCHEMA = "neural-forge-lfm2-weight-complete-basis-v1"
_LFM2_MODULE = "transformers.models.lfm2.modeling_lfm2"


class Lfm2WeightCompleteBasisError(ValueError):
    """The LFM2 topology or its learned-parameter coverage is unsupported."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2WeightCompleteBasisError(
            f"{name} must be a positive integer."
        )
    return value


def _require_parameter(value: Any, path: str) -> torch.nn.Parameter:
    if (
        not isinstance(value, torch.nn.Parameter)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        raise Lfm2WeightCompleteBasisError(
            f"{path} must be a finite dense real floating Parameter."
        )
    return value


def _require_lfm2_rms_norm(
    value: Any,
    *,
    width: int,
    path: str,
) -> torch.nn.Parameter:
    value_type = type(value)
    if (
        value_type.__name__ != "Lfm2RMSNorm"
        or value_type.__module__ != _LFM2_MODULE
    ):
        actual = f"{value_type.__module__}.{value_type.__name__}"
        raise Lfm2WeightCompleteBasisError(
            f"{path} must be {_LFM2_MODULE}.Lfm2RMSNorm, received {actual}."
        )
    weight = _require_parameter(getattr(value, "weight", None), f"{path}.weight")
    if tuple(weight.shape) != (width,):
        raise Lfm2WeightCompleteBasisError(
            f"{path}.weight must have shape [{width}]."
        )
    return weight


def _require_matrix(
    value: Any,
    *,
    shape: tuple[int, int],
    path: str,
) -> torch.nn.Parameter:
    weight = _require_parameter(getattr(value, "weight", None), f"{path}.weight")
    if tuple(weight.shape) != shape:
        raise Lfm2WeightCompleteBasisError(
            f"{path}.weight must have shape {list(shape)}."
        )
    return weight


def _named_parameter_alias_groups(
    model: torch.nn.Module,
) -> tuple[
    dict[str, torch.nn.Parameter],
    dict[str, tuple[str, ...]],
    dict[str, torch.nn.Parameter],
]:
    try:
        named_values = tuple(
            model.named_parameters(recurse=True, remove_duplicate=False)
        )
    except TypeError as exc:  # pragma: no cover - supported PyTorch invariant
        raise Lfm2WeightCompleteBasisError(
            "PyTorch named_parameters(remove_duplicate=False) is required "
            "to audit tied LFM2 aliases."
        ) from exc
    by_path: dict[str, torch.nn.Parameter] = {}
    paths_by_identity: dict[int, list[str]] = {}
    values_by_identity: dict[int, torch.nn.Parameter] = {}
    for name, parameter in named_values:
        if name in by_path:
            raise Lfm2WeightCompleteBasisError(
                f"Duplicate learned-parameter path {name!r}."
            )
        by_path[name] = _require_parameter(parameter, name)
        identity = id(parameter)
        paths_by_identity.setdefault(identity, []).append(name)
        values_by_identity[identity] = parameter
    if not by_path:
        raise Lfm2WeightCompleteBasisError(
            "An LFM2 parent must expose learned parameters."
        )
    aliases_by_path: dict[str, tuple[str, ...]] = {}
    canonical_values: dict[str, torch.nn.Parameter] = {}
    for identity, paths in paths_by_identity.items():
        aliases = tuple(sorted(paths))
        canonical = aliases[0]
        canonical_values[canonical] = values_by_identity[identity]
        for path in aliases:
            aliases_by_path[path] = aliases
    return by_path, aliases_by_path, canonical_values


@dataclass(frozen=True, slots=True)
class Lfm2WeightCompleteInventory:
    """Immutable operators plus an alias-complete coverage manifest."""

    source_width: int
    operators: tuple[ResidualFacingOperator, ...]
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        source_width = _positive_int(self.source_width, "source_width")
        operators = tuple(self.operators)
        if (
            not operators
            or any(
                not isinstance(value, ResidualFacingOperator)
                for value in operators
            )
            or any(value.source_width != source_width for value in operators)
            or not isinstance(self.manifest, Mapping)
        ):
            raise Lfm2WeightCompleteBasisError(
                "The LFM2 weight inventory invariants are invalid."
            )
        object.__setattr__(self, "source_width", source_width)
        object.__setattr__(self, "operators", operators)
        object.__setattr__(
            self,
            "manifest",
            MappingProxyType(dict(self.manifest)),
        )

    def as_dict(self) -> dict[str, Any]:
        return dict(self.manifest)


@dataclass(frozen=True, slots=True)
class Lfm2WeightCompleteBasisResult:
    """Generic basis result bound to the exact LFM2 inventory that built it."""

    inventory: Lfm2WeightCompleteInventory
    basis_result: WeightCompleteBasisResult
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.inventory, Lfm2WeightCompleteInventory)
            or not isinstance(self.basis_result, WeightCompleteBasisResult)
            or self.basis_result.source_width != self.inventory.source_width
            or not isinstance(self.report, Mapping)
        ):
            raise Lfm2WeightCompleteBasisError(
                "The LFM2 weight-complete basis result is inconsistent."
            )
        object.__setattr__(
            self,
            "report",
            MappingProxyType(dict(self.report)),
        )

    @property
    def basis(self) -> torch.Tensor:
        return self.basis_result.basis

    @property
    def full_weight_gram(self) -> torch.Tensor:
        return self.basis_result.full_weight_gram

    @property
    def balanced_structural_gram(self) -> torch.Tensor:
        return self.basis_result.balanced_structural_gram

    @property
    def combined_gram(self) -> torch.Tensor:
        return self.basis_result.combined_gram

    @property
    def source_width(self) -> int:
        return self.basis_result.source_width

    @property
    def target_width(self) -> int:
        return self.basis_result.target_width

    @property
    def map_sha256(self) -> str:
        return self.basis_result.map_sha256

    def encode(self, source_rows: torch.Tensor) -> torch.Tensor:
        return self.basis_result.encode(source_rows)

    def decode(self, target_rows: torch.Tensor) -> torch.Tensor:
        return self.basis_result.decode(target_rows)

    def apply(self, source_rows: torch.Tensor) -> torch.Tensor:
        return self.basis_result.apply(source_rows)

    def __call__(self, source_rows: torch.Tensor) -> torch.Tensor:
        return self.basis_result(source_rows)

    def fold_consumer_weight(
        self,
        source_weight: torch.Tensor,
    ) -> torch.Tensor:
        return self.basis_result.fold_consumer_weight(source_weight)

    def fold_producer_weight(
        self,
        source_weight: torch.Tensor,
    ) -> torch.Tensor:
        return self.basis_result.fold_producer_weight(source_weight)

    def materialize_dense(
        self,
        *,
        max_elements: int,
        dtype: torch.dtype | None = None,
        device: Any = "cpu",
    ) -> torch.Tensor:
        return self.basis_result.materialize_dense(
            max_elements=max_elements,
            dtype=dtype,
            device=device,
        )

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


class _InventoryBuilder:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        source_width: int,
    ) -> None:
        (
            self.by_path,
            self.aliases_by_path,
            self.canonical_values,
        ) = _named_parameter_alias_groups(model)
        self.source_width = source_width
        self.operators: list[ResidualFacingOperator] = []
        self.operator_sources: list[dict[str, Any]] = []
        self.dispositions: dict[str, dict[str, Any]] = {}

    def _group(self, path: str) -> tuple[str, tuple[str, ...], torch.nn.Parameter]:
        try:
            aliases = self.aliases_by_path[path]
            parameter = self.by_path[path]
        except KeyError as exc:
            raise Lfm2WeightCompleteBasisError(
                f"Expected learned parameter {path!r} is missing."
            ) from exc
        canonical = aliases[0]
        if self.canonical_values[canonical] is not parameter:
            raise Lfm2WeightCompleteBasisError(
                f"Parameter alias audit disagrees for {path!r}."
            )
        return canonical, aliases, parameter

    def add_operator(
        self,
        *,
        path: str,
        role: str,
        name: str,
        orientation: FeatureOrientation,
        feature_axis: int,
        diagonal: bool = False,
        tensor: torch.Tensor | None = None,
        transform: str = "identity",
        expected_aliases: tuple[str, ...] | None = None,
    ) -> None:
        canonical, aliases, parameter = self._group(path)
        if expected_aliases is None:
            expected_aliases = (path,)
        if aliases != tuple(sorted(expected_aliases)):
            raise Lfm2WeightCompleteBasisError(
                f"Unexpected parameter aliases for {path!r}: {list(aliases)}."
            )
        if canonical in self.dispositions:
            raise Lfm2WeightCompleteBasisError(
                f"Learned parameter alias group {list(aliases)} was classified "
                "more than once."
            )
        operator_tensor = parameter if tensor is None else tensor
        operator = ResidualFacingOperator(
            role=role,
            name=name,
            tensor=operator_tensor,
            orientation=orientation,
            feature_axis=feature_axis,
            diagonal=diagonal,
        )
        if operator.source_width != self.source_width:
            raise Lfm2WeightCompleteBasisError(
                f"{name} exposes width {operator.source_width}, expected "
                f"residual width {self.source_width}."
            )
        self.operators.append(operator)
        source = {
            "operator_name": operator.name,
            "operator_role": operator.role,
            "operator_sha256": operator.operator_sha256,
            "parameter_canonical_path": canonical,
            "parameter_paths": list(aliases),
            "parameter_shape": [int(value) for value in parameter.shape],
            "parameter_dtype": str(parameter.dtype),
            "parameter_numel": int(parameter.numel()),
            "transform": transform,
            "residual_feature_axis": operator.feature_axis,
            "orientation": operator.orientation.value,
            "diagonal": operator.diagonal,
        }
        self.operator_sources.append(source)
        self.dispositions[canonical] = {
            "disposition": "residual_facing_operator",
            "operator_name": operator.name,
            "operator_role": operator.role,
            "reason": "parameter has a proven LFM2 residual/channel axis",
        }

    def exclude_head_local(
        self,
        *,
        path: str,
        layer_index: int,
        kind: str,
        head_dim: int,
    ) -> None:
        canonical, aliases, parameter = self._group(path)
        if aliases != (path,):
            raise Lfm2WeightCompleteBasisError(
                f"Unexpected parameter aliases for {path!r}: {list(aliases)}."
            )
        if canonical in self.dispositions:
            raise Lfm2WeightCompleteBasisError(
                f"Learned parameter {path!r} was classified more than once."
            )
        if tuple(parameter.shape) != (head_dim,):
            raise Lfm2WeightCompleteBasisError(
                f"{path} must have the attention head-local width {head_dim}."
            )
        self.dispositions[canonical] = {
            "disposition": "excluded_non_residual_axis",
            "operator_name": None,
            "operator_role": None,
            "reason": (
                f"attention {kind} RMSNorm is head-local (axis={head_dim}), "
                f"not model-residual (axis={self.source_width})"
            ),
            "layer_index": layer_index,
        }

    def finish(
        self,
        *,
        layer_kinds: tuple[str, ...],
        tied_embedding_head: bool,
    ) -> Lfm2WeightCompleteInventory:
        unclassified = sorted(
            set(self.canonical_values).difference(self.dispositions)
        )
        if unclassified:
            raise Lfm2WeightCompleteBasisError(
                "Unclassified LFM2 learned parameter alias groups: "
                + ", ".join(unclassified)
                + "."
            )
        alias_groups: list[dict[str, Any]] = []
        included = 0
        excluded = 0
        for canonical in sorted(self.canonical_values):
            parameter = self.canonical_values[canonical]
            aliases = self.aliases_by_path[canonical]
            disposition = self.dispositions[canonical]
            if disposition["disposition"] == "residual_facing_operator":
                included += 1
            else:
                excluded += 1
            alias_groups.append(
                {
                    "canonical_path": canonical,
                    "paths": list(aliases),
                    "alias_count": len(aliases),
                    "shape": [int(value) for value in parameter.shape],
                    "dtype": str(parameter.dtype),
                    "numel": int(parameter.numel()),
                    **disposition,
                }
            )
        operators = tuple(
            sorted(
                self.operators,
                key=lambda value: (
                    value.role,
                    value.name,
                    value.operator_sha256,
                ),
            )
        )
        operator_sources = sorted(
            self.operator_sources,
            key=lambda value: (
                value["operator_role"],
                value["operator_name"],
            ),
        )
        if included != len(operators):
            raise Lfm2WeightCompleteBasisError(
                "Every included parameter alias group must own exactly one "
                "residual-facing operator."
            )
        operator_names = [value.name for value in operators]
        if len(operator_names) != len(set(operator_names)):
            raise Lfm2WeightCompleteBasisError(
                "LFM2 residual-facing operator names must be unique."
            )
        tied_groups = [
            value for value in alias_groups if value["alias_count"] > 1
        ]
        expected_tied_groups = 1 if tied_embedding_head else 0
        if (
            len(tied_groups) != expected_tied_groups
            or any(
                set(value["paths"])
                != {"model.embed_tokens.weight", "lm_head.weight"}
                for value in tied_groups
            )
        ):
            raise Lfm2WeightCompleteBasisError(
                "Only the configured embedding/LM-head alias may share an "
                "LFM2 learned Parameter."
            )
        manifest_body: dict[str, Any] = {
            "schema": LFM2_WEIGHT_INVENTORY_SCHEMA,
            "family": "lfm2",
            "implementation": _LFM2_MODULE,
            "source_width": self.source_width,
            "layer_count": len(layer_kinds),
            "layer_kinds": list(layer_kinds),
            "operator_count": len(operators),
            "operator_sources": operator_sources,
            "parameter_alias_groups": alias_groups,
            "coverage": {
                "learned_parameter_path_count": len(self.by_path),
                "unique_parameter_alias_group_count": len(alias_groups),
                "residual_facing_alias_group_count": included,
                "excluded_non_residual_alias_group_count": excluded,
                "unclassified_alias_group_count": 0,
                "each_residual_facing_alias_group_represented_once": True,
                "each_learned_alias_group_classified_once": True,
                "tied_embedding_head": tied_embedding_head,
                "tied_embedding_head_counted_once": True,
                "all_layers_structurally_inspected": True,
            },
            "prompt_independent_weight_objective": True,
            "model_forward_calls": 0,
            "trace_capture_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "autograd_used": False,
            "filesystem_writes": 0,
            "hashes": {
                "operator_manifest_sha256": _canonical_sha256(
                    [
                        {
                            "role": value.role,
                            "name": value.name,
                            "sha256": value.operator_sha256,
                        }
                        for value in operators
                    ]
                ),
                "alias_coverage_sha256": _canonical_sha256(alias_groups),
            },
        }
        manifest_body["inventory_sha256"] = _canonical_sha256(manifest_body)
        return Lfm2WeightCompleteInventory(
            source_width=self.source_width,
            operators=operators,
            manifest=manifest_body,
        )


def inventory_lfm2_weight_complete_operators(
    parent_model: torch.nn.Module,
) -> Lfm2WeightCompleteInventory:
    """Inventory every residual-facing learned LFM2 parameter exactly once."""

    if not isinstance(parent_model, torch.nn.Module):
        raise Lfm2WeightCompleteBasisError(
            "parent_model must be a torch.nn.Module."
        )
    config = getattr(parent_model, "config", None)
    if getattr(config, "model_type", None) != "lfm2":
        raise Lfm2WeightCompleteBasisError(
            "parent_model.config.model_type must be lfm2."
        )
    source_width = _positive_int(
        getattr(config, "hidden_size", None),
        "config.hidden_size",
    )
    body = getattr(parent_model, "model", None)
    layers_value = getattr(body, "layers", None)
    if body is None or layers_value is None:
        raise Lfm2WeightCompleteBasisError(
            "LFM2 must expose model.layers."
        )
    try:
        layers = tuple(layers_value)
    except TypeError as exc:
        raise Lfm2WeightCompleteBasisError(
            "LFM2 model.layers must be iterable."
        ) from exc
    expected_depth = _positive_int(
        getattr(config, "num_hidden_layers", None),
        "config.num_hidden_layers",
    )
    if len(layers) != expected_depth:
        raise Lfm2WeightCompleteBasisError(
            "config.num_hidden_layers disagrees with model.layers."
        )
    embed_tokens = getattr(body, "embed_tokens", None)
    lm_head = getattr(parent_model, "lm_head", None)
    vocab_size = _positive_int(
        getattr(config, "vocab_size", None),
        "config.vocab_size",
    )
    embedding_weight = _require_matrix(
        embed_tokens,
        shape=(vocab_size, source_width),
        path="model.embed_tokens",
    )
    head_weight = _require_matrix(
        lm_head,
        shape=(vocab_size, source_width),
        path="lm_head",
    )
    if getattr(lm_head, "bias", None) is not None:
        raise Lfm2WeightCompleteBasisError("LFM2 lm_head.bias must be absent.")
    tied = getattr(config, "tie_word_embeddings", None)
    if not isinstance(tied, bool):
        raise Lfm2WeightCompleteBasisError(
            "config.tie_word_embeddings must be boolean."
        )
    if tied != (embedding_weight is head_weight):
        raise Lfm2WeightCompleteBasisError(
            "Configured LFM2 embedding/head tying disagrees with Parameter aliasing."
        )

    builder = _InventoryBuilder(
        model=parent_model,
        source_width=source_width,
    )
    if tied:
        builder.add_operator(
            path="model.embed_tokens.weight",
            role="lexical_interface",
            name="lexical.tied_embedding_head.weight",
            orientation=FeatureOrientation.OUTPUT,
            feature_axis=1,
            expected_aliases=(
                "lm_head.weight",
                "model.embed_tokens.weight",
            ),
            transform="tied embedding output span and LM-head input sensitivity",
        )
    else:
        builder.add_operator(
            path="model.embed_tokens.weight",
            role="lexical_interface",
            name="lexical.embedding.weight",
            orientation=FeatureOrientation.OUTPUT,
            feature_axis=1,
            transform="embedding output span",
        )
        builder.add_operator(
            path="lm_head.weight",
            role="lexical_interface",
            name="lexical.lm_head.weight",
            orientation=FeatureOrientation.INPUT,
            feature_axis=1,
            transform="LM-head input sensitivity",
        )

    layer_kinds: list[str] = []
    for layer_index, layer in enumerate(layers):
        contract = inspect_lfm2_layer(layer)
        if (
            contract.layer_index != layer_index
            or contract.hidden_size != source_width
        ):
            raise Lfm2WeightCompleteBasisError(
                f"LFM2 layer {layer_index} contract disagrees with model order "
                "or residual width."
            )
        layer_kinds.append(contract.kind)
        prefix = f"model.layers.{layer_index}"
        builder.add_operator(
            path=f"{prefix}.operator_norm.weight",
            role="residual_normalization",
            name=f"layer.{layer_index:03d}.operator_norm.weight",
            orientation=FeatureOrientation.OUTPUT,
            feature_axis=0,
            diagonal=True,
        )
        builder.add_operator(
            path=f"{prefix}.ffn_norm.weight",
            role="residual_normalization",
            name=f"layer.{layer_index:03d}.ffn_norm.weight",
            orientation=FeatureOrientation.OUTPUT,
            feature_axis=0,
            diagonal=True,
        )
        mlp = layer.feed_forward
        builder.add_operator(
            path=f"{prefix}.feed_forward.w1.weight",
            role="swiglu_input_sensitivity",
            name=f"layer.{layer_index:03d}.swiglu.gate.weight",
            orientation=FeatureOrientation.INPUT,
            feature_axis=1,
            transform="SiLU gate input sensitivity",
        )
        builder.add_operator(
            path=f"{prefix}.feed_forward.w3.weight",
            role="swiglu_input_sensitivity",
            name=f"layer.{layer_index:03d}.swiglu.up.weight",
            orientation=FeatureOrientation.INPUT,
            feature_axis=1,
            transform="multiplicative up-branch input sensitivity",
        )
        builder.add_operator(
            path=f"{prefix}.feed_forward.w2.weight",
            role="swiglu_output_span",
            name=f"layer.{layer_index:03d}.swiglu.down.weight",
            orientation=FeatureOrientation.OUTPUT,
            feature_axis=0,
            transform="post-SwiGLU residual output span",
        )

        if contract.kind == "full_attention":
            attention_contract = contract.operator
            if not isinstance(attention_contract, Lfm2AttentionContract):
                raise Lfm2WeightCompleteBasisError(
                    "Canonical LFM2 attention contract type mismatch."
                )
            attention = layer.self_attn
            for projection, semantic in (
                ("q_proj", "query"),
                ("k_proj", "key"),
                ("v_proj", "value"),
            ):
                builder.add_operator(
                    path=f"{prefix}.self_attn.{projection}.weight",
                    role="attention_qkv_input_sensitivity",
                    name=(
                        f"layer.{layer_index:03d}.attention."
                        f"{semantic}.weight"
                    ),
                    orientation=FeatureOrientation.INPUT,
                    feature_axis=1,
                )
            builder.add_operator(
                path=f"{prefix}.self_attn.out_proj.weight",
                role="attention_output_span",
                name=f"layer.{layer_index:03d}.attention.output.weight",
                orientation=FeatureOrientation.OUTPUT,
                feature_axis=0,
            )
            for norm_name, semantic in (
                ("q_layernorm", "query"),
                ("k_layernorm", "key"),
            ):
                norm = getattr(attention, norm_name)
                _require_lfm2_rms_norm(
                    norm,
                    width=attention_contract.head_dim,
                    path=f"{prefix}.self_attn.{norm_name}",
                )
                builder.exclude_head_local(
                    path=f"{prefix}.self_attn.{norm_name}.weight",
                    layer_index=layer_index,
                    kind=semantic,
                    head_dim=attention_contract.head_dim,
                )
        elif contract.kind == "short_conv":
            conv_contract = contract.operator
            if not isinstance(conv_contract, Lfm2ShortConvContract):
                raise Lfm2WeightCompleteBasisError(
                    "Canonical LFM2 short-conv contract type mismatch."
                )
            operator = layer.conv
            builder.add_operator(
                path=f"{prefix}.conv.in_proj.weight",
                role="short_conv_input_sensitivity",
                name=f"layer.{layer_index:03d}.short_conv.in_proj.weight",
                orientation=FeatureOrientation.INPUT,
                feature_axis=1,
                transform="coupled B/C/x producer input sensitivity",
            )
            builder.add_operator(
                path=f"{prefix}.conv.out_proj.weight",
                role="short_conv_output_span",
                name=f"layer.{layer_index:03d}.short_conv.out_proj.weight",
                orientation=FeatureOrientation.OUTPUT,
                feature_axis=0,
                transform="gated FIR consumer residual output span",
            )
            builder.add_operator(
                path=f"{prefix}.conv.conv.weight",
                role="short_conv_state_channel",
                name=f"layer.{layer_index:03d}.short_conv.fir.weight",
                orientation=FeatureOrientation.OUTPUT,
                feature_axis=0,
                transform=(
                    "depthwise causal FIR/state channel axis; stored lag order "
                    "preserved"
                ),
            )
            if conv_contract.bias:
                in_bias = _require_parameter(
                    operator.in_proj.bias,
                    f"{prefix}.conv.in_proj.bias",
                )
                builder.add_operator(
                    path=f"{prefix}.conv.in_proj.bias",
                    role="short_conv_state_channel",
                    name=f"layer.{layer_index:03d}.short_conv.in_proj.bias",
                    orientation=FeatureOrientation.OUTPUT,
                    feature_axis=1,
                    tensor=in_bias.reshape(3, source_width),
                    transform="reshape [3H] bias to [B/C/x,H] channel coordinates",
                )
                builder.add_operator(
                    path=f"{prefix}.conv.conv.bias",
                    role="short_conv_state_channel",
                    name=f"layer.{layer_index:03d}.short_conv.fir.bias",
                    orientation=FeatureOrientation.OUTPUT,
                    feature_axis=0,
                    diagonal=True,
                    transform="depthwise FIR channel offset",
                )
                builder.add_operator(
                    path=f"{prefix}.conv.out_proj.bias",
                    role="short_conv_output_span",
                    name=f"layer.{layer_index:03d}.short_conv.out_proj.bias",
                    orientation=FeatureOrientation.OUTPUT,
                    feature_axis=0,
                    diagonal=True,
                    transform="residual output offset",
                )
        else:  # pragma: no cover - canonical contract is exhaustive
            raise Lfm2WeightCompleteBasisError(
                f"Unsupported canonical LFM2 layer kind {contract.kind!r}."
            )

    final_norm = getattr(body, "embedding_norm", None)
    _require_lfm2_rms_norm(
        final_norm,
        width=source_width,
        path="model.embedding_norm",
    )
    builder.add_operator(
        path="model.embedding_norm.weight",
        role="residual_normalization",
        name="output.final_norm.weight",
        orientation=FeatureOrientation.OUTPUT,
        feature_axis=0,
        diagonal=True,
    )
    return builder.finish(
        layer_kinds=tuple(layer_kinds),
        tied_embedding_head=tied,
    )


def build_lfm2_weight_complete_basis(
    parent_model: torch.nn.Module,
    target_width: int,
    *,
    activation_gram: torch.Tensor | None = None,
    activation_weight: float = 0.0,
    maximum_activation_weight: float = 0.25,
    minimum_full_weight_retained_energy_fraction: float = 0.0,
    require_all_source_axis_loading: bool = True,
    source_axis_loading_tolerance: float = 1e-18,
    rank_tolerance: float = 1e-12,
    degeneracy_relative_tolerance: float = 1e-11,
    degeneracy_absolute_tolerance: float = 1e-13,
    enforce_coverage_gate: bool = True,
) -> Lfm2WeightCompleteBasisResult:
    """Build an auditable prompt-independent LFM2 residual coordinate basis."""

    inventory = inventory_lfm2_weight_complete_operators(parent_model)
    basis_result = build_weight_complete_basis(
        inventory.operators,
        target_width,
        activation_gram=activation_gram,
        activation_weight=activation_weight,
        maximum_activation_weight=maximum_activation_weight,
        minimum_full_weight_retained_energy_fraction=(
            minimum_full_weight_retained_energy_fraction
        ),
        require_all_source_axis_loading=require_all_source_axis_loading,
        source_axis_loading_tolerance=source_axis_loading_tolerance,
        rank_tolerance=rank_tolerance,
        degeneracy_relative_tolerance=degeneracy_relative_tolerance,
        degeneracy_absolute_tolerance=degeneracy_absolute_tolerance,
        enforce_coverage_gate=enforce_coverage_gate,
    )
    report: dict[str, Any] = {
        "schema": LFM2_WEIGHT_BASIS_SCHEMA,
        "family": "lfm2",
        "inventory": inventory.as_dict(),
        "basis": basis_result.as_dict(),
        "prompt_independent_weight_objective": True,
        "bounded_activation_supplement": (
            basis_result.report["activation_supplement"]
        ),
        "model_forward_calls": 0,
        "trace_capture_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "autograd_used": False,
        "filesystem_writes": 0,
    }
    report["run_sha256"] = _canonical_sha256(report)
    return Lfm2WeightCompleteBasisResult(
        inventory=inventory,
        basis_result=basis_result,
        report=report,
    )


__all__ = [
    "LFM2_WEIGHT_BASIS_SCHEMA",
    "LFM2_WEIGHT_INVENTORY_SCHEMA",
    "Lfm2WeightCompleteBasisError",
    "Lfm2WeightCompleteBasisResult",
    "Lfm2WeightCompleteInventory",
    "build_lfm2_weight_complete_basis",
    "inventory_lfm2_weight_complete_operators",
]
