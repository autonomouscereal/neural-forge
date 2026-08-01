"""Automatic model-to-causal-circuit graph compilation.

Inspectors are plugins selected by proven topology.  Family names are useful
diagnostics, but the emitted graph is expressed only in canonical circuit,
edge, parameter-role, cache/state, token, and modality concepts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .dynamic_circuit_ir import (
    CircuitEdge,
    CircuitKind,
    CircuitNode,
    CircuitPort,
    DynamicCircuitIRError,
    EdgeKind,
    ModalityInterface,
    ModelCircuitGraph,
    ParameterSpec,
    PortKind,
    canonical_modality,
    canonical_sha256,
)
from .hf_tokenizer_bridge import (
    CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY,
    PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY,
    TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY,
    TOKENIZER_BRIDGE_CONFIG_KEYS,
    TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY,
)


class DynamicModelInspectionError(DynamicCircuitIRError):
    """No inspector can prove a complete graph for the supplied model."""


def _singleton_config_scalar(value: Any, name: str) -> Any:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(value)
        if len(values) != 1:
            raise DynamicModelInspectionError(
                f"{name} must be a scalar or exact singleton sequence."
            )
        return values[0]
    return value


class ModelCircuitInspector(Protocol):
    name: str
    priority: int

    def supports(self, model: Any) -> bool: ...

    def inspect(
        self,
        model: Any,
        *,
        model_name: str,
        tokenizer_fingerprint: str | None = None,
    ) -> ModelCircuitGraph: ...


def _qualified_class(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _config_payload(model: Any) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if config is None:
        raise DynamicModelInspectionError("Model has no configuration object.")
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
    else:
        value = vars(config)
    if not isinstance(value, Mapping):
        raise DynamicModelInspectionError("Model configuration is not a mapping.")

    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                str(key): normalize(nested)
                for key, nested in sorted(item.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            return [normalize(nested) for nested in item]
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        return str(item)

    return normalize(value)


def _storage_groups(model: Any) -> dict[str, str]:
    groups: dict[tuple[Any, ...], str] = {}
    result: dict[str, str] = {}
    for name, parameter in sorted(
        model.named_parameters(recurse=True, remove_duplicate=False)
    ):
        if getattr(parameter, "device", None) is None:
            raise DynamicModelInspectionError("Model parameter has no device.")
        if parameter.device.type == "meta":
            raise DynamicModelInspectionError(
                "Dynamic inspection requires materialized parameters."
            )
        try:
            key = (
                parameter.device.type,
                parameter.device.index,
                str(parameter.dtype),
                int(parameter.untyped_storage().data_ptr()),
                int(parameter.storage_offset()),
                tuple(int(value) for value in parameter.shape),
                tuple(int(value) for value in parameter.stride()),
            )
        except (AttributeError, RuntimeError) as exc:
            raise DynamicModelInspectionError(
                f"Could not establish parameter storage identity for {name!r}."
            ) from exc
        group = groups.setdefault(key, f"storage-{len(groups):06d}")
        result[name] = group
    return result


def _parameter_counts(model: Any) -> tuple[int, int, int]:
    logical = 0
    unique: dict[str, tuple[int, int]] = {}
    groups = _storage_groups(model)
    for name, value in model.named_parameters(recurse=True, remove_duplicate=False):
        count = int(value.numel())
        size = int(value.element_size())
        logical += count
        group = groups[name]
        existing = unique.get(group)
        if existing is not None and existing != (count, size):
            raise DynamicModelInspectionError(
                "Aliased parameter storage has inconsistent shape or dtype."
            )
        unique[group] = (count, size)
    return (
        logical,
        sum(value[0] for value in unique.values()),
        sum(value[0] * value[1] for value in unique.values()),
    )


def _module_path(model: Any, target: Any) -> str:
    for path, module in model.named_modules():
        if module is target:
            if not path:
                raise DynamicModelInspectionError(
                    "The requested operator is the model root, not a child path."
                )
            return path
    raise DynamicModelInspectionError(
        f"Could not resolve module path for {_qualified_class(target)}."
    )


def _module_parameters(
    model: Any,
    module_path: str,
    role_prefix: str,
    groups: Mapping[str, str],
    captured: set[str],
) -> tuple[ParameterSpec, ...]:
    try:
        module = model.get_submodule(module_path)
    except (AttributeError, KeyError) as exc:
        raise DynamicModelInspectionError(
            f"Could not resolve module path {module_path!r}."
        ) from exc
    values = []
    for local_name, parameter in module.named_parameters(
        recurse=True, remove_duplicate=False
    ):
        path = f"{module_path}.{local_name}" if local_name else module_path
        if path in captured:
            raise DynamicModelInspectionError(
                f"Parameter {path!r} belongs to multiple circuit nodes."
            )
        captured.add(path)
        role = f"{role_prefix}.{local_name.replace('.', '_')}"
        values.append(
            ParameterSpec(
                path=path,
                role=role,
                shape=tuple(int(value) for value in parameter.shape),
                dtype=str(parameter.dtype),
                storage_group=groups[path],
                trainable=bool(parameter.requires_grad),
            )
        )
    return tuple(values)


def _port(
    name: str,
    kind: PortKind | str,
    *dimensions: int | str,
    modality: str | None = None,
    stateful: bool = False,
) -> CircuitPort:
    return CircuitPort(
        name=name,
        kind=kind,
        dimensions=tuple(dimensions),
        modality=modality,
        stateful=stateful,
    )


@dataclass(slots=True)
class _GraphBuilder:
    model: Any
    model_name: str
    family: str
    tokenizer_fingerprint: str | None
    groups: Mapping[str, str]
    nodes: list[CircuitNode]
    edges: list[CircuitEdge]
    interfaces: list[ModalityInterface]
    captured: set[str]
    order: int = 0

    def add_node(
        self,
        node_id: str,
        kind: CircuitKind | str,
        path: str,
        *,
        layer_index: int | None,
        inputs: tuple[CircuitPort, ...],
        outputs: tuple[CircuitPort, ...],
        role_prefix: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> CircuitNode:
        parameters = (
            ()
            if role_prefix is None
            else _module_parameters(
                self.model,
                path,
                role_prefix,
                self.groups,
                self.captured,
            )
        )
        node = CircuitNode(
            node_id=node_id,
            kind=kind,
            order=self.order,
            path=path,
            layer_index=layer_index,
            inputs=inputs,
            outputs=outputs,
            parameters=parameters,
            attributes=tuple((attributes or {}).items()),
        )
        self.nodes.append(node)
        self.order += 1
        return node

    def edge(
        self,
        source: CircuitNode,
        source_port: str,
        target: CircuitNode,
        target_port: str,
        kind: EdgeKind | str,
        *,
        recurrent: bool = False,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        self.edges.append(
            CircuitEdge(
                source_node=source.node_id,
                source_port=source_port,
                target_node=target.node_id,
                target_port=target_port,
                kind=kind,
                recurrent=recurrent,
                attributes=tuple((attributes or {}).items()),
            )
        )

    def finish(self, *, metadata: Mapping[str, Any] | None = None) -> ModelCircuitGraph:
        expected = {
            name
            for name, _value in self.model.named_parameters(
                recurse=True, remove_duplicate=False
            )
        }
        if self.captured != expected:
            missing = sorted(expected - self.captured)
            extra = sorted(self.captured - expected)
            raise DynamicModelInspectionError(
                "Inspector did not establish exact parameter ownership; "
                f"missing={missing[:16]!r}, extra={extra[:16]!r}."
            )
        logical, unique, unique_bytes = _parameter_counts(self.model)
        return ModelCircuitGraph(
            model_name=self.model_name,
            family=self.family,
            model_class=_qualified_class(self.model),
            nodes=tuple(self.nodes),
            edges=tuple(self.edges),
            interfaces=tuple(self.interfaces),
            logical_parameter_count=logical,
            unique_parameter_count=unique,
            unique_parameter_bytes=unique_bytes,
            config_fingerprint=canonical_sha256(_config_payload(self.model)),
            tokenizer_fingerprint=self.tokenizer_fingerprint,
            metadata=tuple((metadata or {}).items()),
        )


class Qwen3CodebookCircuitInspector:
    """Compile a full-depth Qwen3 runtime backed by global codebooks."""

    name = "qwen3-codebook"
    priority = 300

    def supports(self, model: Any) -> bool:
        return (
            getattr(getattr(model, "config", None), "model_type", None) == "qwen3"
            and hasattr(model, "attention_codebook")
            and hasattr(model, "neuron_codebook")
            and hasattr(model, "parent_token_rows")
            and hasattr(model, "neural_forge_codebook_report")
        )

    def inspect(
        self,
        model: Any,
        *,
        model_name: str,
        tokenizer_fingerprint: str | None = None,
    ) -> ModelCircuitGraph:
        try:
            import torch
            from torch import nn
            from transformers import Qwen3ForCausalLM

            from .hf_attention_codebook import (
                AttentionCodebookProjectionView,
                AttentionCodebookRegistry,
                GlobalAttentionCodebook,
                RaggedAttentionCodebookRegistry,
            )
            from .hf_neuron_codebook import (
                GlobalNeuronCodebook,
                NeuronCodebookLayerView,
                NeuronCodebookRegistry,
            )
            from .hf_qwen_codebook_model import qwen3_codebook_parameter_report
        except (ImportError, AttributeError) as exc:
            raise DynamicModelInspectionError(
                "Qwen3 codebook inspection requires torch, Transformers Qwen3, "
                "and the neural-forge codebook runtime."
            ) from exc

        if not isinstance(model, Qwen3ForCausalLM):
            raise DynamicModelInspectionError(
                "A Qwen3 codebook graph requires Qwen3ForCausalLM."
            )
        try:
            report = qwen3_codebook_parameter_report(model)
        except Exception as exc:
            raise DynamicModelInspectionError(
                "The Qwen3 codebook runtime failed its ownership/accounting contract."
            ) from exc

        config = model.config
        backbone = getattr(model, "model", None)
        layers = getattr(backbone, "layers", None)
        embedding = model.get_input_embeddings()
        head = model.get_output_embeddings()
        position_module = getattr(backbone, "rotary_emb", None)
        final_norm_module = getattr(backbone, "norm", None)
        attention_codebook = getattr(model, "attention_codebook", None)
        neuron_codebook = getattr(model, "neuron_codebook", None)
        parent_token_rows = getattr(model, "parent_token_rows", None)
        if (
            not isinstance(backbone, nn.Module)
            or not isinstance(layers, nn.ModuleList)
            or not isinstance(embedding, nn.Embedding)
            or not isinstance(head, nn.Linear)
            or not isinstance(position_module, nn.Module)
            or not isinstance(final_norm_module, nn.Module)
            or not isinstance(attention_codebook, GlobalAttentionCodebook)
            or not isinstance(neuron_codebook, GlobalNeuronCodebook)
            or not isinstance(parent_token_rows, torch.Tensor)
        ):
            raise DynamicModelInspectionError(
                "The Qwen3 codebook runtime is missing a required concrete component."
            )
        attention_registry = attention_codebook.registry
        neuron_registry = neuron_codebook.registry
        if not isinstance(
            attention_registry,
            (AttentionCodebookRegistry, RaggedAttentionCodebookRegistry),
        ) or not isinstance(neuron_registry, NeuronCodebookRegistry):
            raise DynamicModelInspectionError(
                "The Qwen3 codebook runtime lacks a supported global registry."
            )

        logical_layers = len(layers)
        configured_layers = getattr(config, "num_hidden_layers", None)
        if (
            isinstance(configured_layers, bool)
            or not isinstance(configured_layers, int)
            or configured_layers != logical_layers
            or logical_layers <= 0
            or len(attention_codebook) != logical_layers
            or len(neuron_codebook) != logical_layers
            or report.logical_layers != logical_layers
            or report.residual_block_count != logical_layers
            or attention_registry.logical_layers != logical_layers
            or neuron_registry.logical_layers != logical_layers
        ):
            raise DynamicModelInspectionError(
                "Qwen3 decoder depth and codebook logical depth disagree."
            )
        if not (
            report.full_logical_depth
            and report.exact_parent_token_rows
            and report.tied_input_output_embeddings
            and report.parameterless_layer_views
            and report.single_registry_owner_per_codebook
            and report.parent_runtime_reference_count == 0
        ):
            raise DynamicModelInspectionError(
                "Qwen3 codebook runtime invariants are incomplete."
            )

        hidden = int(report.hidden_size)
        intermediate = int(report.intermediate_size)
        vocabulary = int(report.target_vocab_size)
        if (
            hidden <= 0
            or intermediate <= 0
            or vocabulary <= 0
            or int(getattr(config, "hidden_size", -1)) != hidden
            or int(getattr(config, "intermediate_size", -1)) != intermediate
            or int(getattr(config, "vocab_size", -1)) != vocabulary
            or embedding.embedding_dim != hidden
            or embedding.num_embeddings != vocabulary
            or head.in_features != hidden
            or head.out_features != vocabulary
            or embedding.weight is not head.weight
            or not bool(getattr(config, "tie_word_embeddings", False))
        ):
            raise DynamicModelInspectionError(
                "Qwen3 codebook dimensions or tied token storage disagree."
            )
        if (
            parent_token_rows.ndim != 1
            or int(parent_token_rows.numel()) != vocabulary
            or parent_token_rows.dtype
            not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }
        ):
            raise DynamicModelInspectionError(
                "parent_token_rows must contain one integer row per target token."
            )
        parent_rows_cpu = parent_token_rows.detach().to(
            device="cpu",
            dtype=torch.int64,
        )
        if (
            int(parent_rows_cpu.min()) < 0
            or int(parent_rows_cpu.max()) >= int(report.source_vocab_size)
            or int(torch.unique(parent_rows_cpu).numel()) != vocabulary
        ):
            raise DynamicModelInspectionError(
                "parent_token_rows is not an exact one-to-one parent vocabulary map."
            )

        for layer_index, layer in enumerate(layers):
            attention = getattr(layer, "self_attn", None)
            layer_attention_view = attention_codebook[layer_index]
            layer_neuron_view = neuron_codebook[layer_index]
            if (
                not isinstance(attention, nn.Module)
                or not isinstance(layer_neuron_view, NeuronCodebookLayerView)
                or getattr(layer, "mlp", None) is not layer_neuron_view
                or any(layer_neuron_view.parameters())
                or layer_neuron_view.state_dict()
            ):
                raise DynamicModelInspectionError(
                    "A Qwen3 logical MLP is not its parameterless codebook view."
                )
            for role in ("q_proj", "k_proj", "v_proj", "o_proj"):
                projection = getattr(attention, role, None)
                if (
                    not isinstance(projection, AttentionCodebookProjectionView)
                    or projection is not getattr(layer_attention_view, role)
                    or any(projection.parameters())
                    or projection.state_dict()
                ):
                    raise DynamicModelInspectionError(
                        "A Qwen3 logical attention projection is not its "
                        "parameterless codebook view."
                    )
            for norm_name in (
                "input_layernorm",
                "post_attention_layernorm",
            ):
                if not isinstance(getattr(layer, norm_name, None), nn.Module):
                    raise DynamicModelInspectionError(
                        f"Qwen3 layer {layer_index} lacks {norm_name}."
                    )
            for norm_name in ("q_norm", "k_norm"):
                if not isinstance(getattr(attention, norm_name, None), nn.Module):
                    raise DynamicModelInspectionError(
                        f"Qwen3 attention layer {layer_index} lacks {norm_name}."
                    )

        registered_logical, unique_parameters, _unique_bytes = _parameter_counts(model)
        if (
            registered_logical != report.registered_logical_parameter_count
            or unique_parameters != report.unique_parameter_count
            or sum(int(value.numel()) for value in attention_registry.parameters())
            != report.attention_unique_parameter_count
            or sum(int(value.numel()) for value in neuron_registry.parameters())
            != report.neuron_unique_parameter_count
        ):
            raise DynamicModelInspectionError(
                "Qwen3 codebook report and live parameter storage disagree."
            )

        layer_types_value = getattr(config, "layer_types", None)
        if layer_types_value is None:
            layer_types = ("full_attention",) * logical_layers
        elif isinstance(layer_types_value, Sequence) and not isinstance(
            layer_types_value, (str, bytes)
        ):
            layer_types = tuple(str(value) for value in layer_types_value)
            if len(layer_types) != logical_layers:
                raise DynamicModelInspectionError(
                    "Qwen3 layer_types does not cover every logical layer."
                )
        else:
            raise DynamicModelInspectionError(
                "Qwen3 layer_types must be a sequence when present."
            )
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_type = "default"
        if isinstance(rope_scaling, Mapping):
            rope_type = str(
                rope_scaling.get("rope_type") or rope_scaling.get("type") or "default"
            )

        builder = _GraphBuilder(
            model=model,
            model_name=model_name,
            family="qwen3_codebook",
            tokenizer_fingerprint=tokenizer_fingerprint,
            groups=_storage_groups(model),
            nodes=[],
            edges=[],
            interfaces=[],
            captured=set(),
        )
        embedding_path = _module_path(model, embedding)
        head_path = _module_path(model, head)
        position_path = _module_path(model, position_module)
        final_norm_path = _module_path(model, final_norm_module)
        attention_registry_path = _module_path(model, attention_registry)
        neuron_registry_path = _module_path(model, neuron_registry)

        embedding_node = builder.add_node(
            "input.token_embedding",
            CircuitKind.TOKEN_EMBEDDING,
            embedding_path,
            layer_index=None,
            inputs=(_port("token_ids", PortKind.TOKEN_IDS, "batch", "sequence"),),
            outputs=(
                _port(
                    "residual",
                    PortKind.RESIDUAL,
                    "batch",
                    "sequence",
                    hidden,
                    modality="text",
                ),
            ),
            role_prefix="embedding",
            attributes={
                "vocabulary_size": vocabulary,
                "source_vocabulary_size": int(report.source_vocab_size),
                "hidden_size": hidden,
                "exact_parent_token_rows": True,
                "parent_token_row_buffer": "parent_token_rows",
                "token_row_fingerprint": report.token_row_fingerprint,
                "tied_to_output_head": True,
            },
        )
        builder.interfaces.append(
            ModalityInterface(
                interface_id="text.tokens.input",
                modality="text",
                direction="input",
                node_id=embedding_node.node_id,
                port_name="token_ids",
                representation="token_ids",
            )
        )
        position_node = builder.add_node(
            "input.position",
            CircuitKind.POSITION_ENCODING,
            position_path,
            layer_index=None,
            inputs=(_port("position_ids", PortKind.POSITION_IDS, "batch", "sequence"),),
            outputs=(_port("position", PortKind.GENERIC, "batch", "sequence", hidden),),
            attributes={
                "kind": "rope",
                "max_positions": int(
                    getattr(config, "max_position_embeddings", 0) or 0
                ),
                "rotary_dim": int(report.head_dim),
                "rope_theta": float(getattr(config, "rope_theta", 10000.0)),
                "rope_type": rope_type,
            },
        )
        attention_owner = builder.add_node(
            "storage.attention_codebook",
            "attention_codebook_registry",
            attention_registry_path,
            layer_index=None,
            inputs=(),
            outputs=(
                _port(
                    "registry",
                    PortKind.GENERIC,
                    "attention_codebook",
                ),
            ),
            role_prefix="attention_codebook",
            attributes={
                "semantic": "attention_codebook_registry",
                "storage_scope": "global",
                "registered_owner": True,
                "logical_layers": logical_layers,
                "query_heads": int(report.query_heads),
                "key_value_heads": int(report.key_value_heads),
                "head_dim": int(report.head_dim),
                "prototype_count": int(attention_registry.prototype_count),
                "prototypes_per_group": int(report.attention_prototypes_per_group),
                "registry_layout": (
                    "ragged"
                    if isinstance(
                        attention_registry,
                        RaggedAttentionCodebookRegistry,
                    )
                    else "rectangular"
                ),
                "coupled_roles": "q_k_v_o",
                "index_buffers_excluded_from_parameter_budget": True,
            },
        )
        neuron_owner = builder.add_node(
            "storage.neuron_codebook",
            "neuron_codebook_registry",
            neuron_registry_path,
            layer_index=None,
            inputs=(),
            outputs=(
                _port(
                    "registry",
                    PortKind.GENERIC,
                    "neuron_codebook",
                ),
            ),
            role_prefix="neuron_codebook",
            attributes={
                "semantic": "neuron_codebook_registry",
                "storage_scope": "global",
                "registered_owner": True,
                "logical_layers": logical_layers,
                "intermediate_size": intermediate,
                "prototype_count": int(neuron_registry.prototype_count),
                "prototypes_per_neuron": report.neuron_prototypes_per_neuron,
                "coupled_roles": "gate_up_post_swiglu_down",
                "index_buffers_excluded_from_parameter_budget": True,
            },
        )

        current = embedding_node
        for layer_index, layer in enumerate(layers):
            prefix = f"layer.{layer_index:03d}"
            layer_path = _module_path(model, layer)
            attention_path = _module_path(model, layer.self_attn)
            mlp_path = _module_path(model, layer.mlp)
            attention_norm = builder.add_node(
                f"{prefix}.attention_norm",
                CircuitKind.NORM,
                _module_path(model, layer.input_layernorm),
                layer_index=layer_index,
                inputs=(
                    _port(
                        "residual",
                        PortKind.RESIDUAL,
                        "batch",
                        "sequence",
                        hidden,
                    ),
                ),
                outputs=(
                    _port(
                        "normalized",
                        PortKind.NORMALIZED_RESIDUAL,
                        "batch",
                        "sequence",
                        hidden,
                    ),
                ),
                role_prefix="attention_norm",
                attributes={
                    "norm_kind": "rms_norm",
                    "epsilon": float(getattr(config, "rms_norm_eps", 1e-6)),
                    "weight_kind": "scale",
                },
            )
            attention_node = builder.add_node(
                f"{prefix}.attention",
                CircuitKind.ATTENTION,
                attention_path,
                layer_index=layer_index,
                inputs=(
                    _port(
                        "residual",
                        PortKind.NORMALIZED_RESIDUAL,
                        "batch",
                        "sequence",
                        hidden,
                    ),
                    _port(
                        "position",
                        PortKind.GENERIC,
                        "batch",
                        "sequence",
                        hidden,
                    ),
                    _port(
                        "mask",
                        PortKind.ATTENTION_MASK,
                        "batch",
                        "sequence",
                        "sequence",
                    ),
                    _port(
                        "cache_in",
                        PortKind.CACHE,
                        "layers",
                        stateful=True,
                    ),
                    _port(
                        "codebook",
                        PortKind.GENERIC,
                        "attention_codebook",
                    ),
                ),
                outputs=(
                    _port(
                        "delta",
                        PortKind.BRANCH_DELTA,
                        "batch",
                        "sequence",
                        hidden,
                    ),
                    _port(
                        "cache_out",
                        PortKind.CACHE,
                        "layers",
                        stateful=True,
                    ),
                ),
                role_prefix="attention",
                attributes={
                    "projection_style": "split_q_k_v_o_codebook",
                    "query_heads": int(report.query_heads),
                    "key_value_heads": int(report.key_value_heads),
                    "head_dim": int(report.head_dim),
                    "grouped_query_attention": (
                        report.query_heads != report.key_value_heads
                    ),
                    "qk_rmsnorm": True,
                    "layer_type": layer_types[layer_index],
                    "sliding_window": getattr(config, "sliding_window", None),
                    "residual_source": "pre_attention",
                    "parameterless_projection_views": True,
                    "projection_storage_owner": attention_owner.node_id,
                    "coupled_roles": "q_k_v_o",
                },
            )
            builder.edge(
                current,
                "residual",
                attention_norm,
                "residual",
                EdgeKind.DATA,
            )
            builder.edge(
                attention_norm,
                "normalized",
                attention_node,
                "residual",
                EdgeKind.DATA,
            )
            builder.edge(
                position_node,
                "position",
                attention_node,
                "position",
                EdgeKind.CONTROL,
            )
            builder.edge(
                attention_owner,
                "registry",
                attention_node,
                "codebook",
                EdgeKind.CONTROL,
            )
            builder.edge(
                attention_node,
                "cache_out",
                attention_node,
                "cache_in",
                EdgeKind.CACHE,
                recurrent=True,
            )
            attention_add = builder.add_node(
                f"{prefix}.attention_residual",
                CircuitKind.RESIDUAL_ADD,
                layer_path,
                layer_index=layer_index,
                inputs=(
                    _port(
                        "base",
                        PortKind.RESIDUAL,
                        "batch",
                        "sequence",
                        hidden,
                    ),
                    _port(
                        "delta",
                        PortKind.BRANCH_DELTA,
                        "batch",
                        "sequence",
                        hidden,
                    ),
                ),
                outputs=(
                    _port(
                        "residual",
                        PortKind.RESIDUAL,
                        "batch",
                        "sequence",
                        hidden,
                    ),
                ),
                attributes={"ordered_branch": "attention"},
            )
            builder.edge(
                current,
                "residual",
                attention_add,
                "base",
                EdgeKind.RESIDUAL,
            )
            builder.edge(
                attention_node,
                "delta",
                attention_add,
                "delta",
                EdgeKind.RESIDUAL,
            )
            mlp_norm = builder.add_node(
                f"{prefix}.mlp_norm",
                CircuitKind.NORM,
                _module_path(model, layer.post_attention_layernorm),
                layer_index=layer_index,
                inputs=(
                    _port(
                        "residual",
                        PortKind.RESIDUAL,
                        "batch",
                        "sequence",
                        hidden,
                    ),
                ),
                outputs=(
                    _port(
                        "normalized",
                        PortKind.NORMALIZED_RESIDUAL,
                        "batch",
                        "sequence",
                        hidden,
                    ),
                ),
                role_prefix="mlp_norm",
                attributes={
                    "norm_kind": "rms_norm",
                    "epsilon": float(getattr(config, "rms_norm_eps", 1e-6)),
                    "weight_kind": "scale",
                },
            )
            mlp_node = builder.add_node(
                f"{prefix}.mlp",
                CircuitKind.GATED_MLP,
                mlp_path,
                layer_index=layer_index,
                inputs=(
                    _port(
                        "residual",
                        PortKind.NORMALIZED_RESIDUAL,
                        "batch",
                        "sequence",
                        hidden,
                    ),
                    _port(
                        "codebook",
                        PortKind.GENERIC,
                        "neuron_codebook",
                    ),
                ),
                outputs=(
                    _port(
                        "delta",
                        PortKind.BRANCH_DELTA,
                        "batch",
                        "sequence",
                        hidden,
                    ),
                ),
                role_prefix="mlp",
                attributes={
                    "intermediate_size": intermediate,
                    "activation": neuron_registry.activation,
                    "projection_layout": "gate_up_down_codebook",
                    "residual_source": "post_attention",
                    "parameterless_layer_view": True,
                    "projection_storage_owner": neuron_owner.node_id,
                    "coupled_roles": "gate_up_post_swiglu_down",
                },
            )
            builder.edge(
                attention_add,
                "residual",
                mlp_norm,
                "residual",
                EdgeKind.DATA,
            )
            builder.edge(
                mlp_norm,
                "normalized",
                mlp_node,
                "residual",
                EdgeKind.DATA,
            )
            builder.edge(
                neuron_owner,
                "registry",
                mlp_node,
                "codebook",
                EdgeKind.CONTROL,
            )
            mlp_add = builder.add_node(
                f"{prefix}.mlp_residual",
                CircuitKind.RESIDUAL_ADD,
                layer_path,
                layer_index=layer_index,
                inputs=(
                    _port(
                        "base",
                        PortKind.RESIDUAL,
                        "batch",
                        "sequence",
                        hidden,
                    ),
                    _port(
                        "delta",
                        PortKind.BRANCH_DELTA,
                        "batch",
                        "sequence",
                        hidden,
                    ),
                ),
                outputs=(
                    _port(
                        "residual",
                        PortKind.RESIDUAL,
                        "batch",
                        "sequence",
                        hidden,
                    ),
                ),
                attributes={"ordered_branch": "mlp"},
            )
            builder.edge(
                attention_add,
                "residual",
                mlp_add,
                "base",
                EdgeKind.RESIDUAL,
            )
            builder.edge(
                mlp_node,
                "delta",
                mlp_add,
                "delta",
                EdgeKind.RESIDUAL,
            )
            current = mlp_add

        final_norm = builder.add_node(
            "output.final_norm",
            CircuitKind.NORM,
            final_norm_path,
            layer_index=None,
            inputs=(
                _port(
                    "residual",
                    PortKind.RESIDUAL,
                    "batch",
                    "sequence",
                    hidden,
                ),
            ),
            outputs=(
                _port(
                    "normalized",
                    PortKind.NORMALIZED_RESIDUAL,
                    "batch",
                    "sequence",
                    hidden,
                ),
            ),
            role_prefix="final_norm",
            attributes={
                "norm_kind": "rms_norm",
                "epsilon": float(getattr(config, "rms_norm_eps", 1e-6)),
                "weight_kind": "scale",
            },
        )
        head_node = builder.add_node(
            "output.lm_head",
            CircuitKind.OUTPUT_HEAD,
            head_path,
            layer_index=None,
            inputs=(
                _port(
                    "residual",
                    PortKind.NORMALIZED_RESIDUAL,
                    "batch",
                    "sequence",
                    hidden,
                ),
            ),
            outputs=(
                _port(
                    "logits",
                    PortKind.LOGITS,
                    "batch",
                    "sequence",
                    vocabulary,
                    modality="text",
                ),
            ),
            role_prefix="output_head",
            attributes={
                "vocabulary_size": vocabulary,
                "source_vocabulary_size": int(report.source_vocab_size),
                "tied_to_embedding": True,
                "embedding_storage_owner": embedding_node.node_id,
                "exact_parent_token_rows": True,
            },
        )
        builder.edge(
            current,
            "residual",
            final_norm,
            "residual",
            EdgeKind.DATA,
        )
        builder.edge(
            final_norm,
            "normalized",
            head_node,
            "residual",
            EdgeKind.DATA,
        )
        builder.interfaces.append(
            ModalityInterface(
                interface_id="text.logits.output",
                modality="text",
                direction="output",
                node_id=head_node.node_id,
                port_name="logits",
                representation="token_logits",
            )
        )
        graph = builder.finish(
            metadata={
                "inspector": self.name,
                "runtime_kind": "global_attention_and_neuron_codebooks",
                "source_architecture_family": "qwen3",
                "position_policy": "rope",
                "layer_count": logical_layers,
                "hidden_size": hidden,
                "intermediate_size": intermediate,
                "query_heads": int(report.query_heads),
                "key_value_heads": int(report.key_value_heads),
                "head_dim": int(report.head_dim),
                "tie_word_embeddings": True,
                "exact_parent_token_rows": True,
                "source_vocabulary_size": int(report.source_vocab_size),
                "target_vocabulary_size": vocabulary,
                "token_row_fingerprint": report.token_row_fingerprint,
                "attention_registry_path": attention_registry_path,
                "neuron_registry_path": neuron_registry_path,
                "attention_unique_parameter_count": (
                    report.attention_unique_parameter_count
                ),
                "neuron_unique_parameter_count": (report.neuron_unique_parameter_count),
                "registered_logical_parameter_count": (
                    report.registered_logical_parameter_count
                ),
                "dense_logical_parameter_count": report.logical_parameter_count,
                "index_buffer_bytes": report.index_buffer_bytes,
                "full_logical_depth": True,
                "single_registry_owner_per_codebook": True,
            }
        )
        if (
            graph.logical_parameter_count != report.registered_logical_parameter_count
            or graph.unique_parameter_count != report.unique_parameter_count
        ):
            raise DynamicModelInspectionError(
                "Finished Qwen3 codebook graph misses its exact parameter budget."
            )
        return graph


NATIVE_AUDIO_LANE_CONFIG_KEY = "neural_forge_native_audio_lane"
NATIVE_AUDIO_LANE_MARKER = "neural-forge-native-audio-lane-v1"
NATIVE_AUDIO_MODEL_TYPE = "native_audio_chat"
NATIVE_AUDIO_FAMILY = "native_audio_chat"
NATIVE_AUDIO_REPRESENTATION = "pcm16_frames"


class NativeAudioCircuitInspector:
    """Compile a learned PCM16 frame encoder plus its native text decoder.

    This is the audio mirror of :class:`NeoNativeVisionCircuitInspector`: the
    waveform enters through one ``MODALITY_INPUT`` node whose input port is a
    raw ``pcm16_frames`` modality port, the encoder result is projected into
    the decoder residual stream, and the text decoder is compiled exactly as
    the vision lane compiles its embedded decoder.  There is no frozen
    external encoder anywhere in the graph: every audio tensor is a learned
    parameter this inspector owns.

    ``supports`` is deliberately keyed on an explicit opt-in marker rather
    than on a model type alone.  A model must set
    ``config.neural_forge_native_audio_lane = NATIVE_AUDIO_LANE_MARKER`` to be
    routed here, so the inspector can never hijack a real audio-capable model
    that has not declared this exact topology.
    """

    name = "native-audio"
    priority = 240

    def supports(self, model: Any) -> bool:
        config = getattr(model, "config", None)
        return bool(
            getattr(config, NATIVE_AUDIO_LANE_CONFIG_KEY, None)
            == NATIVE_AUDIO_LANE_MARKER
            and getattr(config, "model_type", None) == NATIVE_AUDIO_MODEL_TYPE
            and hasattr(model, "audio_encoder")
            and hasattr(model, "language_model")
        )

    def inspect(
        self,
        model: Any,
        *,
        model_name: str,
        tokenizer_fingerprint: str | None = None,
    ) -> ModelCircuitGraph:
        try:
            from torch import nn
        except ImportError as exc:  # pragma: no cover - optional torch boundary
            raise DynamicModelInspectionError(
                "Native audio inspection requires PyTorch."
            ) from exc

        audio_modality = canonical_modality("audio", "native audio input modality")
        text_modality = canonical_modality("text", "native audio text modality")
        config = getattr(model, "config", None)
        audio_config = getattr(config, "audio_config", None)
        decoder_config = getattr(config, "decoder_config", None)
        audio_encoder = getattr(model, "audio_encoder", None)
        language_model = getattr(model, "language_model", None)
        if (
            config is None
            or getattr(config, NATIVE_AUDIO_LANE_CONFIG_KEY, None)
            != NATIVE_AUDIO_LANE_MARKER
            or getattr(config, "model_type", None) != NATIVE_AUDIO_MODEL_TYPE
            or audio_config is None
            or decoder_config is None
            or audio_encoder is None
            or language_model is None
        ):
            raise DynamicModelInspectionError(
                "Native audio inspection requires one marked native_audio_chat "
                "config with audio_config, decoder_config, audio_encoder, and "
                "language_model."
            )

        frame_encoder = getattr(audio_encoder, "frame_conv", None)
        frame_projection = getattr(audio_encoder, "projection", None)
        if not isinstance(frame_encoder, nn.Conv1d) or not isinstance(
            frame_projection, nn.Linear
        ):
            raise DynamicModelInspectionError(
                "Native audio front end must expose a Conv1d frame_conv and a "
                "Linear projection."
            )

        try:
            sample_rate = int(
                _singleton_config_scalar(
                    audio_config.sample_rate,
                    "native audio sample_rate",
                )
            )
            frame_size = int(
                _singleton_config_scalar(
                    audio_config.frame_size,
                    "native audio frame_size",
                )
            )
            input_channels = int(
                _singleton_config_scalar(
                    audio_config.num_channels,
                    "native audio num_channels",
                )
            )
            audio_hidden = int(audio_config.hidden_size)
            decoder_hidden = int(decoder_config.hidden_size)
        except DynamicModelInspectionError:
            raise
        except (AttributeError, TypeError, ValueError) as exc:
            raise DynamicModelInspectionError(
                "Native audio front-end/decoder dimensions are incomplete."
            ) from exc
        if (
            sample_rate <= 0
            or frame_size <= 0
            or input_channels <= 0
            or audio_hidden <= 0
            or decoder_hidden <= 0
        ):
            raise DynamicModelInspectionError(
                "Native audio front-end/decoder dimensions must be positive."
            )
        sample_encoding = getattr(audio_config, "sample_encoding", None)
        if sample_encoding != "int16_normalized_pcm":
            raise DynamicModelInspectionError(
                "Native audio front end must consume int16-normalized PCM "
                "samples rather than a precomputed external feature."
            )
        if (
            frame_encoder.in_channels != input_channels
            or frame_encoder.out_channels != audio_hidden
            or frame_encoder.kernel_size != (frame_size,)
            or frame_encoder.stride != (frame_size,)
        ):
            raise DynamicModelInspectionError(
                "Native audio frame encoder differs from the declared strided "
                "PCM frame topology."
            )
        if (
            tuple(frame_projection.weight.shape) != (decoder_hidden, audio_hidden)
            or frame_projection.bias is not None
        ):
            raise DynamicModelInspectionError(
                "Native audio projection must be a bias-free Linear from the "
                "encoder width to the decoder hidden size."
            )

        expected_layers = int(getattr(decoder_config, "num_hidden_layers", 0))
        intermediate_size = int(getattr(decoder_config, "intermediate_size", 0))
        query_heads = int(getattr(decoder_config, "num_attention_heads", 0))
        key_value_heads = int(getattr(decoder_config, "num_key_value_heads", 0))
        head_dim = int(getattr(decoder_config, "head_dim", 0))
        vocab_size = int(getattr(decoder_config, "vocab_size", 0))
        max_positions = int(getattr(decoder_config, "max_position_embeddings", 0))
        if any(
            value <= 0
            for value in (
                expected_layers,
                intermediate_size,
                query_heads,
                key_value_heads,
                head_dim,
                vocab_size,
                max_positions,
            )
        ):
            raise DynamicModelInspectionError(
                "Native audio decoder topology dimensions must be positive."
            )
        if (
            query_heads % key_value_heads
            or query_heads * head_dim != decoder_hidden
            or getattr(decoder_config, "hidden_act", None) != "silu"
            or getattr(decoder_config, "attention_bias", None) is not False
        ):
            raise DynamicModelInspectionError(
                "Native audio decoder must be a bias-free SwiGLU grouped-query "
                "stack whose query projection spans the residual width."
            )

        decoder_core = getattr(language_model, "model", None)
        token_embedding = getattr(decoder_core, "embed_tokens", None)
        position_embedding = getattr(decoder_core, "position_embedding", None)
        decoder_layers = getattr(decoder_core, "layers", None)
        final_norm_module = getattr(decoder_core, "norm", None)
        lm_head_module = getattr(language_model, "lm_head", None)
        if (
            decoder_core is None
            or not isinstance(token_embedding, nn.Embedding)
            or not isinstance(position_embedding, nn.Embedding)
            or not isinstance(decoder_layers, nn.ModuleList)
            or final_norm_module is None
            or not isinstance(lm_head_module, nn.Linear)
        ):
            raise DynamicModelInspectionError(
                "Native audio decoder is missing its canonical token embedding, "
                "learned position embedding, layers, final norm, or output head."
            )
        if len(decoder_layers) != expected_layers:
            raise DynamicModelInspectionError(
                "Native audio decoder depth differs from decoder_config."
            )
        if (
            tuple(token_embedding.weight.shape) != (vocab_size, decoder_hidden)
            or tuple(position_embedding.weight.shape)
            != (max_positions, decoder_hidden)
            or tuple(lm_head_module.weight.shape) != (vocab_size, decoder_hidden)
            or lm_head_module.bias is not None
        ):
            raise DynamicModelInspectionError(
                "Native audio token/position embedding or output head shapes "
                "are incompatible."
            )
        if bool(getattr(decoder_config, "tie_word_embeddings", False)) and (
            token_embedding.weight is not lm_head_module.weight
        ):
            raise DynamicModelInspectionError(
                "Native audio config requests tied token/output weights but "
                "runtime storage is not tied."
            )

        groups = _storage_groups(model)
        builder = _GraphBuilder(
            model=model,
            model_name=model_name,
            family=NATIVE_AUDIO_FAMILY,
            tokenizer_fingerprint=tokenizer_fingerprint,
            groups=groups,
            nodes=[],
            edges=[],
            interfaces=[],
            captured=set(),
        )
        audio_path = _module_path(model, audio_encoder)
        frame_path = _module_path(model, frame_encoder)
        projection_path = _module_path(model, frame_projection)
        language_path = _module_path(model, language_model)

        frames = builder.add_node(
            "input.audio.pcm16_frame_encoder",
            CircuitKind.MODALITY_INPUT,
            frame_path,
            layer_index=None,
            inputs=(
                _port(
                    "waveform",
                    PortKind.MODALITY,
                    "batch",
                    "samples",
                    modality=audio_modality,
                ),
            ),
            outputs=(
                _port(
                    "frames",
                    PortKind.MODALITY,
                    "batch",
                    "audio_frames",
                    audio_hidden,
                    modality=audio_modality,
                ),
            ),
            role_prefix="audio_frame_encoder",
            attributes={
                "input_channels": input_channels,
                "frame_size": frame_size,
                "frame_stride": frame_size,
                "sample_rate": sample_rate,
                "sample_encoding": "int16_normalized_pcm",
                "audio_hidden_size": audio_hidden,
                "activation": "gelu",
                "external_frozen_encoder": False,
            },
        )
        audio_projection = builder.add_node(
            "input.audio.dense_projection",
            CircuitKind.INTERFACE_PROJECTION,
            projection_path,
            layer_index=None,
            inputs=(
                _port(
                    "frames",
                    PortKind.MODALITY,
                    "batch",
                    "audio_frames",
                    audio_hidden,
                    modality=audio_modality,
                ),
            ),
            outputs=(
                _port(
                    "audio_embeddings",
                    PortKind.MODALITY,
                    "audio_tokens",
                    decoder_hidden,
                    modality=audio_modality,
                ),
            ),
            role_prefix="audio_dense_embedding",
            attributes={
                "decoder_hidden_size": decoder_hidden,
                "audio_hidden_size": audio_hidden,
            },
        )
        builder.edge(
            frames,
            "frames",
            audio_projection,
            "frames",
            EdgeKind.MODALITY,
        )
        builder.interfaces.append(
            ModalityInterface(
                interface_id="audio.pcm16_frames.input",
                modality=audio_modality,
                direction="input",
                node_id=frames.node_id,
                port_name="waveform",
                representation=NATIVE_AUDIO_REPRESENTATION,
            )
        )

        captured_audio = {
            path for path in builder.captured if path.startswith(f"{audio_path}.")
        }
        expected_audio = {
            path
            for path, _parameter in model.named_parameters(
                recurse=True,
                remove_duplicate=False,
            )
            if path.startswith(f"{audio_path}.")
        }
        if captured_audio != expected_audio:
            raise DynamicModelInspectionError(
                "Native audio front end contains learned parameters outside "
                "its frame encoder and dense projection operators."
            )

        def require_linear(
            module: Any,
            shape: tuple[int, int],
            label: str,
        ) -> None:
            if (
                not isinstance(module, nn.Linear)
                or tuple(module.weight.shape) != shape
                or module.bias is not None
            ):
                raise DynamicModelInspectionError(
                    f"Native audio decoder {label} must be a bias-free "
                    f"Linear{shape}."
                )

        def require_norm(module: Any, width: int, label: str) -> None:
            parameters = tuple(
                module.named_parameters(
                    recurse=True,
                    remove_duplicate=False,
                )
            )
            if (
                len(parameters) != 1
                or parameters[0][0] != "weight"
                or tuple(parameters[0][1].shape) != (width,)
            ):
                raise DynamicModelInspectionError(
                    f"Native audio decoder {label} must own one width-{width} "
                    "norm weight."
                )

        embedding_path = _module_path(model, token_embedding)
        position_path = _module_path(model, position_embedding)
        final_norm_path = _module_path(model, final_norm_module)
        lm_head_path = _module_path(model, lm_head_module)
        token_node = builder.add_node(
            "decoder.input.token_embedding",
            CircuitKind.TOKEN_EMBEDDING,
            embedding_path,
            layer_index=None,
            inputs=(
                _port(
                    "token_ids",
                    PortKind.TOKEN_IDS,
                    "batch",
                    "sequence",
                ),
            ),
            outputs=(
                _port(
                    "residual",
                    PortKind.RESIDUAL,
                    "batch",
                    "sequence",
                    decoder_hidden,
                    modality=text_modality,
                ),
            ),
            role_prefix="decoder_embedding",
            attributes={
                "vocabulary_size": vocab_size,
                "hidden_size": decoder_hidden,
            },
        )
        builder.interfaces.append(
            ModalityInterface(
                interface_id="text.tokens.input",
                modality=text_modality,
                direction="input",
                node_id=token_node.node_id,
                port_name="token_ids",
                representation="token_ids",
            )
        )
        position = builder.add_node(
            "decoder.input.position",
            CircuitKind.POSITION_ENCODING,
            position_path,
            layer_index=None,
            inputs=(
                _port(
                    "indexes",
                    PortKind.POSITION_IDS,
                    "batch",
                    "sequence",
                ),
            ),
            outputs=(
                _port(
                    "positions",
                    PortKind.RESIDUAL,
                    "batch",
                    "sequence",
                    decoder_hidden,
                ),
            ),
            role_prefix="decoder_position_embedding",
            attributes={
                "kind": "learned_absolute_position_embedding",
                "max_positions": max_positions,
                "hidden_size": decoder_hidden,
            },
        )
        fusion = builder.add_node(
            "input.audio_text_embedding_fusion",
            CircuitKind.INTERFACE_PROJECTION,
            language_path,
            layer_index=None,
            inputs=(
                _port(
                    "text_residual",
                    PortKind.RESIDUAL,
                    "batch",
                    "sequence",
                    decoder_hidden,
                    modality=text_modality,
                ),
                _port(
                    "position_embeddings",
                    PortKind.RESIDUAL,
                    "batch",
                    "sequence",
                    decoder_hidden,
                ),
                _port(
                    "audio_embeddings",
                    PortKind.MODALITY,
                    "audio_tokens",
                    decoder_hidden,
                    modality=audio_modality,
                ),
            ),
            outputs=(
                _port(
                    "residual",
                    PortKind.RESIDUAL,
                    "batch",
                    "sequence",
                    decoder_hidden,
                ),
            ),
            attributes={
                "operation": "audio_context_tokens_in_decoder_residual_stream",
                "learned_parameters": 0,
            },
        )
        builder.edge(
            token_node,
            "residual",
            fusion,
            "text_residual",
            EdgeKind.TOKEN,
        )
        builder.edge(
            position,
            "positions",
            fusion,
            "position_embeddings",
            EdgeKind.DATA,
        )
        builder.edge(
            audio_projection,
            "audio_embeddings",
            fusion,
            "audio_embeddings",
            EdgeKind.MODALITY,
        )

        current = fusion
        current_port = "residual"
        query_width = query_heads * head_dim
        key_value_width = key_value_heads * head_dim
        for layer_index, layer in enumerate(decoder_layers):
            attention_norm_module = getattr(layer, "input_layernorm", None)
            attention_module = getattr(layer, "self_attn", None)
            mlp_norm_module = getattr(layer, "post_attention_layernorm", None)
            mlp_module = getattr(layer, "mlp", None)
            if any(
                value is None
                for value in (
                    attention_norm_module,
                    attention_module,
                    mlp_norm_module,
                    mlp_module,
                )
            ):
                raise DynamicModelInspectionError(
                    f"Native audio decoder layer {layer_index} is missing "
                    "canonical operators."
                )
            require_linear(
                getattr(attention_module, "q_proj", None),
                (query_width, decoder_hidden),
                f"layer {layer_index} q_proj",
            )
            require_linear(
                getattr(attention_module, "k_proj", None),
                (key_value_width, decoder_hidden),
                f"layer {layer_index} k_proj",
            )
            require_linear(
                getattr(attention_module, "v_proj", None),
                (key_value_width, decoder_hidden),
                f"layer {layer_index} v_proj",
            )
            require_linear(
                getattr(attention_module, "o_proj", None),
                (decoder_hidden, query_width),
                f"layer {layer_index} o_proj",
            )
            require_linear(
                getattr(mlp_module, "gate_proj", None),
                (intermediate_size, decoder_hidden),
                f"layer {layer_index} gate_proj",
            )
            require_linear(
                getattr(mlp_module, "up_proj", None),
                (intermediate_size, decoder_hidden),
                f"layer {layer_index} up_proj",
            )
            require_linear(
                getattr(mlp_module, "down_proj", None),
                (decoder_hidden, intermediate_size),
                f"layer {layer_index} down_proj",
            )
            require_norm(
                attention_norm_module,
                decoder_hidden,
                f"layer {layer_index} input_layernorm",
            )
            require_norm(
                mlp_norm_module,
                decoder_hidden,
                f"layer {layer_index} post_attention_layernorm",
            )
            if {
                name
                for name, _parameter in attention_module.named_parameters(
                    recurse=True,
                    remove_duplicate=False,
                )
            } != {
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "o_proj.weight",
            }:
                raise DynamicModelInspectionError(
                    f"Native audio decoder layer {layer_index} attention owns "
                    "unsupported learned tensors."
                )
            if {
                name
                for name, _parameter in mlp_module.named_parameters(
                    recurse=True,
                    remove_duplicate=False,
                )
            } != {
                "gate_proj.weight",
                "up_proj.weight",
                "down_proj.weight",
            }:
                raise DynamicModelInspectionError(
                    f"Native audio decoder layer {layer_index} MLP owns "
                    "unsupported learned tensors."
                )

            layer_path = _module_path(model, layer)
            attention_norm = builder.add_node(
                f"decoder.layer.{layer_index:03d}.attention_norm",
                CircuitKind.NORM,
                _module_path(model, attention_norm_module),
                layer_index=layer_index,
                inputs=(
                    _port(
                        "residual",
                        PortKind.RESIDUAL,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                ),
                outputs=(
                    _port(
                        "normalized",
                        PortKind.NORMALIZED_RESIDUAL,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                ),
                role_prefix="decoder_attention_norm",
                attributes={
                    "norm_kind": "rms_norm",
                    "epsilon": float(decoder_config.rms_norm_eps),
                },
            )
            attention = builder.add_node(
                f"decoder.layer.{layer_index:03d}.attention",
                CircuitKind.ATTENTION,
                _module_path(model, attention_module),
                layer_index=layer_index,
                inputs=(
                    _port(
                        "residual",
                        PortKind.NORMALIZED_RESIDUAL,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                    _port(
                        "mask",
                        PortKind.ATTENTION_MASK,
                        "batch",
                        "sequence",
                        "sequence",
                    ),
                    _port(
                        "cache_in",
                        PortKind.CACHE,
                        "layers",
                        stateful=True,
                    ),
                ),
                outputs=(
                    _port(
                        "delta",
                        PortKind.BRANCH_DELTA,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                    _port(
                        "cache_out",
                        PortKind.CACHE,
                        "layers",
                        stateful=True,
                    ),
                ),
                role_prefix="decoder_attention",
                attributes={
                    "projection_style": "split_qkv",
                    "query_heads": query_heads,
                    "key_value_heads": key_value_heads,
                    "head_dim": head_dim,
                    "grouped_query_attention": True,
                    "layer_type": "full_attention",
                },
            )
            attention_add = builder.add_node(
                f"decoder.layer.{layer_index:03d}.attention_residual",
                CircuitKind.RESIDUAL_ADD,
                layer_path,
                layer_index=layer_index,
                inputs=(
                    _port(
                        "base",
                        PortKind.RESIDUAL,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                    _port(
                        "delta",
                        PortKind.BRANCH_DELTA,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                ),
                outputs=(
                    _port(
                        "residual",
                        PortKind.RESIDUAL,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                ),
                attributes={"ordered_branch": "attention"},
            )
            mlp_norm = builder.add_node(
                f"decoder.layer.{layer_index:03d}.mlp_norm",
                CircuitKind.NORM,
                _module_path(model, mlp_norm_module),
                layer_index=layer_index,
                inputs=(
                    _port(
                        "residual",
                        PortKind.RESIDUAL,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                ),
                outputs=(
                    _port(
                        "normalized",
                        PortKind.NORMALIZED_RESIDUAL,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                ),
                role_prefix="decoder_mlp_norm",
                attributes={
                    "norm_kind": "rms_norm",
                    "epsilon": float(decoder_config.rms_norm_eps),
                },
            )
            mlp = builder.add_node(
                f"decoder.layer.{layer_index:03d}.mlp",
                CircuitKind.GATED_MLP,
                _module_path(model, mlp_module),
                layer_index=layer_index,
                inputs=(
                    _port(
                        "residual",
                        PortKind.NORMALIZED_RESIDUAL,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                ),
                outputs=(
                    _port(
                        "delta",
                        PortKind.BRANCH_DELTA,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                ),
                role_prefix="decoder_mlp",
                attributes={
                    "intermediate_size": intermediate_size,
                    "activation": "silu",
                    "projection_layout": "output_input",
                },
            )
            mlp_add = builder.add_node(
                f"decoder.layer.{layer_index:03d}.mlp_residual",
                CircuitKind.RESIDUAL_ADD,
                layer_path,
                layer_index=layer_index,
                inputs=(
                    _port(
                        "base",
                        PortKind.RESIDUAL,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                    _port(
                        "delta",
                        PortKind.BRANCH_DELTA,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                ),
                outputs=(
                    _port(
                        "residual",
                        PortKind.RESIDUAL,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                ),
                attributes={"ordered_branch": "mlp"},
            )
            builder.edge(
                current,
                current_port,
                attention_norm,
                "residual",
                EdgeKind.DATA,
            )
            builder.edge(
                attention_norm,
                "normalized",
                attention,
                "residual",
                EdgeKind.DATA,
            )
            builder.edge(
                attention,
                "cache_out",
                attention,
                "cache_in",
                EdgeKind.CACHE,
                recurrent=True,
            )
            builder.edge(
                current,
                current_port,
                attention_add,
                "base",
                EdgeKind.RESIDUAL,
            )
            builder.edge(
                attention,
                "delta",
                attention_add,
                "delta",
                EdgeKind.RESIDUAL,
            )
            builder.edge(
                attention_add,
                "residual",
                mlp_norm,
                "residual",
                EdgeKind.DATA,
            )
            builder.edge(
                mlp_norm,
                "normalized",
                mlp,
                "residual",
                EdgeKind.DATA,
            )
            builder.edge(
                attention_add,
                "residual",
                mlp_add,
                "base",
                EdgeKind.RESIDUAL,
            )
            builder.edge(
                mlp,
                "delta",
                mlp_add,
                "delta",
                EdgeKind.RESIDUAL,
            )
            current = mlp_add
            current_port = "residual"

        require_norm(final_norm_module, decoder_hidden, "final norm")
        final_norm = builder.add_node(
            "decoder.output.final_norm",
            CircuitKind.NORM,
            final_norm_path,
            layer_index=None,
            inputs=(
                _port(
                    "residual",
                    PortKind.RESIDUAL,
                    "batch",
                    "sequence",
                    decoder_hidden,
                ),
            ),
            outputs=(
                _port(
                    "normalized",
                    PortKind.NORMALIZED_RESIDUAL,
                    "batch",
                    "sequence",
                    decoder_hidden,
                ),
            ),
            role_prefix="decoder_final_norm",
            attributes={
                "norm_kind": "rms_norm",
                "epsilon": float(decoder_config.rms_norm_eps),
            },
        )
        head = builder.add_node(
            "decoder.output.lm_head",
            CircuitKind.OUTPUT_HEAD,
            lm_head_path,
            layer_index=None,
            inputs=(
                _port(
                    "residual",
                    PortKind.NORMALIZED_RESIDUAL,
                    "batch",
                    "sequence",
                    decoder_hidden,
                ),
            ),
            outputs=(
                _port(
                    "logits",
                    PortKind.LOGITS,
                    "batch",
                    "sequence",
                    vocab_size,
                    modality=text_modality,
                ),
            ),
            role_prefix="decoder_output_head",
            attributes={
                "vocabulary_size": vocab_size,
                "tied_to_embedding": bool(
                    getattr(decoder_config, "tie_word_embeddings", False)
                ),
            },
        )
        builder.edge(
            current,
            current_port,
            final_norm,
            "residual",
            EdgeKind.DATA,
        )
        builder.edge(
            final_norm,
            "normalized",
            head,
            "residual",
            EdgeKind.DATA,
        )
        builder.interfaces.append(
            ModalityInterface(
                interface_id="text.logits.output",
                modality=text_modality,
                direction="output",
                node_id=head.node_id,
                port_name="logits",
                representation="token_logits",
            )
        )

        audio_parameter_count = sum(
            int(parameter.numel())
            for path, parameter in model.named_parameters(
                recurse=True,
                remove_duplicate=False,
            )
            if path.startswith(f"{audio_path}.")
        )
        return builder.finish(
            metadata={
                "inspector": self.name,
                "decoder_family": "native_audio_text_decoder",
                "layer_count": expected_layers,
                "hidden_size": decoder_hidden,
                "decoder_intermediate_size": intermediate_size,
                "audio_hidden_size": audio_hidden,
                "audio_parameter_count": audio_parameter_count,
                "audio_sample_rate": sample_rate,
                "audio_frame_size": frame_size,
                "native_pcm16_frame_encoder": True,
                "audio_token_replacement_fusion": True,
                "external_frozen_audio_encoder": False,
            }
        )


class NeoNativeVisionCircuitInspector:
    """Compile NEO's learned RGB tower and embedded dense Qwen3 decoder."""

    name = "neo-native-vision"
    priority = 250

    def supports(self, model: Any) -> bool:
        config = getattr(model, "config", None)
        return bool(
            getattr(config, "model_type", None) == "neo_chat"
            and hasattr(model, "vision_model")
            and hasattr(model, "language_model")
        )

    def inspect(
        self,
        model: Any,
        *,
        model_name: str,
        tokenizer_fingerprint: str | None = None,
    ) -> ModelCircuitGraph:
        try:
            from torch import nn
        except ImportError as exc:  # pragma: no cover - optional HF boundary
            raise DynamicModelInspectionError(
                "NEO native-vision inspection requires PyTorch."
            ) from exc

        config = getattr(model, "config", None)
        vision_config = getattr(config, "vision_config", None)
        llm_config = getattr(config, "llm_config", None)
        vision_model = getattr(model, "vision_model", None)
        language_model = getattr(model, "language_model", None)
        if (
            config is None
            or getattr(config, "model_type", None) != "neo_chat"
            or vision_config is None
            or llm_config is None
            or vision_model is None
            or language_model is None
        ):
            raise DynamicModelInspectionError(
                "NEO inspection requires one neo_chat config with vision_config, "
                "llm_config, vision_model, and language_model."
            )
        if getattr(getattr(language_model, "config", None), "model_type", None) != (
            "qwen3"
        ):
            raise DynamicModelInspectionError(
                "NEO native vision requires an embedded dense Qwen3 language model."
            )
        if getattr(llm_config, "model_type", None) != "qwen3":
            raise DynamicModelInspectionError(
                "NEO llm_config must serialize the embedded Qwen3 decoder."
            )

        embeddings = getattr(vision_model, "embeddings", None)
        patch_embedding = getattr(embeddings, "patch_embedding", None)
        dense_embedding = getattr(embeddings, "dense_embedding", None)
        if not isinstance(patch_embedding, nn.Conv2d) or not isinstance(
            dense_embedding, nn.Conv2d
        ):
            raise DynamicModelInspectionError(
                "NEO vision must expose Conv2d patch_embedding and dense_embedding."
            )

        try:
            patch_size = int(vision_config.patch_size)
            num_channels = int(vision_config.num_channels)
            vision_hidden = int(vision_config.hidden_size)
            projected_hidden = int(
                _singleton_config_scalar(
                    vision_config.llm_hidden_size,
                    "NEO vision llm_hidden_size",
                )
            )
            downsample_ratio = float(
                _singleton_config_scalar(
                    vision_config.downsample_ratio,
                    "NEO vision downsample_ratio",
                )
            )
            decoder_hidden = int(language_model.config.hidden_size)
        except DynamicModelInspectionError:
            raise
        except (AttributeError, TypeError, ValueError) as exc:
            raise DynamicModelInspectionError(
                "NEO vision/decoder dimensions are incomplete."
            ) from exc
        if (
            patch_size <= 0
            or num_channels <= 0
            or vision_hidden <= 0
            or projected_hidden <= 0
            or decoder_hidden <= 0
            or not 0.0 < downsample_ratio <= 1.0
        ):
            raise DynamicModelInspectionError(
                "NEO vision/decoder dimensions must be positive and finite."
            )
        downsample_factor = int(round(1.0 / downsample_ratio))
        if (
            downsample_factor <= 0
            or abs((1.0 / downsample_ratio) - downsample_factor) > 1e-9
        ):
            raise DynamicModelInspectionError(
                "NEO downsample_ratio must have an integral reciprocal."
            )
        if projected_hidden != decoder_hidden:
            raise DynamicModelInspectionError(
                "NEO dense vision output does not match decoder hidden size."
            )
        if (
            patch_embedding.in_channels != num_channels
            or patch_embedding.out_channels != vision_hidden
            or patch_embedding.kernel_size != (patch_size, patch_size)
            or patch_embedding.stride != (patch_size, patch_size)
        ):
            raise DynamicModelInspectionError(
                "NEO patch embedding differs from the declared raw-patch topology."
            )
        if (
            dense_embedding.in_channels != vision_hidden
            or dense_embedding.out_channels != decoder_hidden
            or dense_embedding.kernel_size != (downsample_factor, downsample_factor)
            or dense_embedding.stride != (downsample_factor, downsample_factor)
        ):
            raise DynamicModelInspectionError(
                "NEO dense embedding differs from the declared image-token topology."
            )

        expected_layers = int(getattr(llm_config, "num_hidden_layers", 0))
        intermediate_size = int(getattr(llm_config, "intermediate_size", 0))
        query_heads = int(getattr(llm_config, "num_attention_heads", 0))
        key_value_heads = int(getattr(llm_config, "num_key_value_heads", 0))
        head_dim = int(getattr(llm_config, "head_dim", 0))
        vocab_size = int(getattr(llm_config, "vocab_size", 0))
        rope_theta_hw = getattr(llm_config, "rope_theta_hw", None)
        max_positions_hw = getattr(
            llm_config,
            "max_position_embeddings_hw",
            None,
        )
        rope_theta = getattr(llm_config, "rope_theta", None)
        if rope_theta is None:
            rope_parameters = getattr(llm_config, "rope_parameters", None)
            if isinstance(rope_parameters, Mapping):
                rope_theta = rope_parameters.get("rope_theta")
        if any(
            value <= 0
            for value in (
                expected_layers,
                intermediate_size,
                query_heads,
                key_value_heads,
                head_dim,
                vocab_size,
            )
        ) or (
            not isinstance(rope_theta, (int, float))
            or rope_theta <= 0
            or not isinstance(rope_theta_hw, (int, float))
            or rope_theta_hw <= 0
            or isinstance(max_positions_hw, bool)
            or not isinstance(max_positions_hw, int)
            or max_positions_hw <= 0
            or head_dim % 2
        ):
            raise DynamicModelInspectionError(
                "NEO Qwen3 topology dimensions must be positive."
            )
        if (
            query_heads <= key_value_heads
            or query_heads % key_value_heads
            or getattr(llm_config, "hidden_act", None) != "silu"
            or getattr(llm_config, "attention_bias", None) is not False
            or getattr(llm_config, "use_sliding_window", None) is not False
        ):
            raise DynamicModelInspectionError(
                "NEO decoder must be bias-free full-attention grouped-query Qwen3."
            )
        if canonical_sha256(_config_payload(language_model)) != canonical_sha256(
            _config_payload(type("_ConfigModel", (), {"config": llm_config})())
        ):
            raise DynamicModelInspectionError(
                "NEO llm_config differs from the embedded language model config."
            )

        decoder_core = getattr(language_model, "model", None)
        token_embedding = getattr(decoder_core, "embed_tokens", None)
        decoder_layers = getattr(decoder_core, "layers", None)
        final_norm_module = getattr(decoder_core, "norm", None)
        lm_head_module = getattr(language_model, "lm_head", None)
        if (
            decoder_core is None
            or not isinstance(token_embedding, nn.Embedding)
            or not isinstance(decoder_layers, nn.ModuleList)
            or final_norm_module is None
            or not isinstance(lm_head_module, nn.Linear)
        ):
            raise DynamicModelInspectionError(
                "NEO Qwen3 decoder is missing its canonical embedding, layers, "
                "rotary position operator, final norm, or output head."
            )
        if len(decoder_layers) != expected_layers:
            raise DynamicModelInspectionError(
                "NEO Qwen3 decoder depth differs from llm_config."
            )
        first_attention = getattr(decoder_layers[0], "self_attn", None)
        decoder_position = getattr(first_attention, "rotary_emb", None)
        if decoder_position is None:
            raise DynamicModelInspectionError(
                "NEO Qwen3 attention is missing its temporal rotary operator."
            )
        if (
            tuple(token_embedding.weight.shape) != (vocab_size, decoder_hidden)
            or tuple(lm_head_module.weight.shape) != (vocab_size, decoder_hidden)
            or token_embedding.padding_idx is not None
            and not 0 <= token_embedding.padding_idx < vocab_size
            or lm_head_module.bias is not None
        ):
            raise DynamicModelInspectionError(
                "NEO Qwen3 token embedding/output head shapes are incompatible."
            )
        if tuple(decoder_position.parameters()):
            raise DynamicModelInspectionError(
                "NEO Qwen3 rotary position operator unexpectedly owns parameters."
            )
        if bool(getattr(llm_config, "tie_word_embeddings", False)) and (
            token_embedding.weight is not lm_head_module.weight
        ):
            raise DynamicModelInspectionError(
                "NEO Qwen3 config requests tied token/output weights but runtime "
                "storage is not tied."
            )

        groups = _storage_groups(model)
        builder = _GraphBuilder(
            model=model,
            model_name=model_name,
            family="neo_chat",
            tokenizer_fingerprint=tokenizer_fingerprint,
            groups=groups,
            nodes=[],
            edges=[],
            interfaces=[],
            captured=set(),
        )
        vision_path = _module_path(model, vision_model)
        embeddings_path = _module_path(model, embeddings)
        patch_path = _module_path(model, patch_embedding)
        dense_path = _module_path(model, dense_embedding)
        language_path = _module_path(model, language_model)

        patch_vector = num_channels * patch_size * patch_size
        patch = builder.add_node(
            "input.image.rgb_patch_embedding",
            CircuitKind.MODALITY_INPUT,
            patch_path,
            layer_index=None,
            inputs=(
                _port(
                    "pixel_values",
                    PortKind.MODALITY,
                    "image_patches",
                    patch_vector,
                    modality="image",
                ),
                _port(
                    "grid_hw",
                    PortKind.GENERIC,
                    "images",
                    2,
                    modality="image",
                ),
            ),
            outputs=(
                _port(
                    "patches",
                    PortKind.MODALITY,
                    "image_patches",
                    vision_hidden,
                    modality="image",
                ),
                _port(
                    "grid_hw",
                    PortKind.GENERIC,
                    "images",
                    2,
                    modality="image",
                ),
            ),
            role_prefix="vision_patch_embedding",
            attributes={
                "input_channels": num_channels,
                "patch_size": patch_size,
                "vision_hidden_size": vision_hidden,
                "activation": "gelu",
            },
        )
        image_position = builder.add_node(
            "input.image.position_2d",
            CircuitKind.POSITION_ENCODING,
            embeddings_path,
            layer_index=None,
            inputs=(
                _port(
                    "patches",
                    PortKind.MODALITY,
                    "image_patches",
                    vision_hidden,
                    modality="image",
                ),
                _port(
                    "grid_hw",
                    PortKind.GENERIC,
                    "images",
                    2,
                    modality="image",
                ),
            ),
            outputs=(
                _port(
                    "positioned_patches",
                    PortKind.MODALITY,
                    "image_patches",
                    vision_hidden,
                    modality="image",
                ),
            ),
            attributes={
                "kind": "rope_2d",
                "vision_hidden_size": vision_hidden,
                "persistent_learned_parameters": 0,
            },
        )
        image_projection = builder.add_node(
            "input.image.dense_projection",
            CircuitKind.INTERFACE_PROJECTION,
            dense_path,
            layer_index=None,
            inputs=(
                _port(
                    "positioned_patches",
                    PortKind.MODALITY,
                    "image_patches",
                    vision_hidden,
                    modality="image",
                ),
            ),
            outputs=(
                _port(
                    "image_embeddings",
                    PortKind.MODALITY,
                    "image_tokens",
                    decoder_hidden,
                    modality="image",
                ),
            ),
            role_prefix="vision_dense_embedding",
            attributes={
                "downsample_factor": downsample_factor,
                "decoder_hidden_size": decoder_hidden,
            },
        )
        builder.edge(
            patch,
            "patches",
            image_position,
            "patches",
            EdgeKind.MODALITY,
        )
        builder.edge(
            patch,
            "grid_hw",
            image_position,
            "grid_hw",
            EdgeKind.MODALITY,
        )
        builder.edge(
            image_position,
            "positioned_patches",
            image_projection,
            "positioned_patches",
            EdgeKind.MODALITY,
        )
        builder.interfaces.extend(
            (
                ModalityInterface(
                    interface_id="image.rgb_patches.input",
                    modality="image",
                    direction="input",
                    node_id=patch.node_id,
                    port_name="pixel_values",
                    representation="normalized_rgb_patch_vectors",
                ),
                ModalityInterface(
                    interface_id="image.grid.input",
                    modality="image",
                    direction="input",
                    node_id=patch.node_id,
                    port_name="grid_hw",
                    representation="patch_grid_height_width",
                ),
            )
        )

        captured_vision = {
            path for path in builder.captured if path.startswith(f"{vision_path}.")
        }
        expected_vision = {
            path
            for path, _parameter in model.named_parameters(
                recurse=True,
                remove_duplicate=False,
            )
            if path.startswith(f"{vision_path}.")
        }
        if captured_vision != expected_vision:
            raise DynamicModelInspectionError(
                "NEO vision contains learned parameters outside its patch and "
                "dense embedding operators."
            )

        def require_linear(
            module: Any,
            shape: tuple[int, int],
            label: str,
        ) -> None:
            if (
                not isinstance(module, nn.Linear)
                or tuple(module.weight.shape) != shape
                or module.bias is not None
            ):
                raise DynamicModelInspectionError(
                    f"NEO Qwen3 {label} must be a bias-free Linear{shape}."
                )

        def require_norm(module: Any, width: int, label: str) -> None:
            parameters = tuple(
                module.named_parameters(
                    recurse=True,
                    remove_duplicate=False,
                )
            )
            if (
                len(parameters) != 1
                or parameters[0][0] != "weight"
                or tuple(parameters[0][1].shape) != (width,)
            ):
                raise DynamicModelInspectionError(
                    f"NEO Qwen3 {label} must own one width-{width} norm weight."
                )

        embedding_path = _module_path(model, token_embedding)
        decoder_position_path = _module_path(model, decoder_position)
        final_norm_path = _module_path(model, final_norm_module)
        lm_head_path = _module_path(model, lm_head_module)
        token_node = builder.add_node(
            "decoder.input.token_embedding",
            CircuitKind.TOKEN_EMBEDDING,
            embedding_path,
            layer_index=None,
            inputs=(
                _port(
                    "token_ids",
                    PortKind.TOKEN_IDS,
                    "batch",
                    "sequence",
                ),
            ),
            outputs=(
                _port(
                    "residual",
                    PortKind.RESIDUAL,
                    "batch",
                    "sequence",
                    decoder_hidden,
                    modality="text",
                ),
            ),
            role_prefix="decoder_embedding",
            attributes={
                "vocabulary_size": vocab_size,
                "hidden_size": decoder_hidden,
            },
        )
        builder.interfaces.append(
            ModalityInterface(
                interface_id="text.tokens.input",
                modality="text",
                direction="input",
                node_id=token_node.node_id,
                port_name="token_ids",
                representation="token_ids",
            )
        )
        fusion = builder.add_node(
            "input.image_text_embedding_fusion",
            CircuitKind.INTERFACE_PROJECTION,
            language_path,
            layer_index=None,
            inputs=(
                _port(
                    "text_residual",
                    PortKind.RESIDUAL,
                    "batch",
                    "sequence",
                    decoder_hidden,
                    modality="text",
                ),
                _port(
                    "image_embeddings",
                    PortKind.MODALITY,
                    "image_tokens",
                    decoder_hidden,
                    modality="image",
                ),
            ),
            outputs=(
                _port(
                    "residual",
                    PortKind.RESIDUAL,
                    "batch",
                    "sequence",
                    decoder_hidden,
                ),
            ),
            attributes={
                "operation": "replace_image_context_token_embeddings",
                "learned_parameters": 0,
            },
        )
        builder.edge(
            token_node,
            "residual",
            fusion,
            "text_residual",
            EdgeKind.TOKEN,
        )
        builder.edge(
            image_projection,
            "image_embeddings",
            fusion,
            "image_embeddings",
            EdgeKind.MODALITY,
        )
        position = builder.add_node(
            "decoder.input.position",
            CircuitKind.POSITION_ENCODING,
            decoder_position_path,
            layer_index=None,
            inputs=(
                _port(
                    "indexes",
                    PortKind.POSITION_IDS,
                    3,
                    "sequence",
                ),
            ),
            outputs=(
                _port(
                    "indexes",
                    PortKind.POSITION_IDS,
                    3,
                    "sequence",
                ),
            ),
            attributes={
                "kind": "temporal_height_width_rope",
                "max_positions": int(llm_config.max_position_embeddings),
                "max_positions_hw": int(max_positions_hw),
                "temporal_rotary_dim": head_dim,
                "spatial_rotary_dim": head_dim // 2,
                "rope_theta": float(rope_theta),
                "rope_theta_hw": float(rope_theta_hw),
            },
        )

        current = fusion
        current_port = "residual"
        query_width = query_heads * head_dim
        key_value_width = key_value_heads * head_dim
        layer_types = getattr(llm_config, "layer_types", None)
        if layer_types is not None and (
            not isinstance(layer_types, (list, tuple))
            or len(layer_types) != expected_layers
            or any(value != "full_attention" for value in layer_types)
        ):
            raise DynamicModelInspectionError(
                "NEO Qwen3 requires full attention in every decoder layer."
            )

        for layer_index, layer in enumerate(decoder_layers):
            attention_norm_module = getattr(layer, "input_layernorm", None)
            attention_module = getattr(layer, "self_attn", None)
            mlp_norm_module = getattr(layer, "post_attention_layernorm", None)
            mlp_module = getattr(layer, "mlp", None)
            if any(
                value is None
                for value in (
                    attention_norm_module,
                    attention_module,
                    mlp_norm_module,
                    mlp_module,
                )
            ):
                raise DynamicModelInspectionError(
                    f"NEO Qwen3 layer {layer_index} is missing canonical operators."
                )
            q_proj = getattr(attention_module, "q_proj", None)
            q_proj_hw = getattr(attention_module, "q_proj_hw", None)
            k_proj = getattr(attention_module, "k_proj", None)
            k_proj_hw = getattr(attention_module, "k_proj_hw", None)
            v_proj = getattr(attention_module, "v_proj", None)
            o_proj = getattr(attention_module, "o_proj", None)
            q_norm = getattr(attention_module, "q_norm", None)
            q_norm_h = getattr(attention_module, "q_norm_h", None)
            q_norm_w = getattr(attention_module, "q_norm_w", None)
            k_norm = getattr(attention_module, "k_norm", None)
            k_norm_h = getattr(attention_module, "k_norm_h", None)
            k_norm_w = getattr(attention_module, "k_norm_w", None)
            temporal_rotary = getattr(attention_module, "rotary_emb", None)
            spatial_rotary = getattr(attention_module, "rotary_emb_hw", None)
            gate_proj = getattr(mlp_module, "gate_proj", None)
            up_proj = getattr(mlp_module, "up_proj", None)
            down_proj = getattr(mlp_module, "down_proj", None)
            require_linear(
                q_proj,
                (query_width, decoder_hidden),
                f"layer {layer_index} q_proj",
            )
            require_linear(
                q_proj_hw,
                (query_width, decoder_hidden),
                f"layer {layer_index} q_proj_hw",
            )
            require_linear(
                k_proj,
                (key_value_width, decoder_hidden),
                f"layer {layer_index} k_proj",
            )
            require_linear(
                k_proj_hw,
                (key_value_width, decoder_hidden),
                f"layer {layer_index} k_proj_hw",
            )
            require_linear(
                v_proj,
                (key_value_width, decoder_hidden),
                f"layer {layer_index} v_proj",
            )
            require_linear(
                o_proj,
                (decoder_hidden, query_width),
                f"layer {layer_index} o_proj",
            )
            require_norm(q_norm, head_dim, f"layer {layer_index} q_norm")
            require_norm(
                q_norm_h,
                head_dim // 2,
                f"layer {layer_index} q_norm_h",
            )
            require_norm(
                q_norm_w,
                head_dim // 2,
                f"layer {layer_index} q_norm_w",
            )
            require_norm(k_norm, head_dim, f"layer {layer_index} k_norm")
            require_norm(
                k_norm_h,
                head_dim // 2,
                f"layer {layer_index} k_norm_h",
            )
            require_norm(
                k_norm_w,
                head_dim // 2,
                f"layer {layer_index} k_norm_w",
            )
            if (
                temporal_rotary is None
                or spatial_rotary is None
                or tuple(temporal_rotary.parameters())
                or tuple(spatial_rotary.parameters())
            ):
                raise DynamicModelInspectionError(
                    f"NEO Qwen3 layer {layer_index} must expose parameter-free "
                    "temporal and height/width rotary operators."
                )
            require_linear(
                gate_proj,
                (intermediate_size, decoder_hidden),
                f"layer {layer_index} gate_proj",
            )
            require_linear(
                up_proj,
                (intermediate_size, decoder_hidden),
                f"layer {layer_index} up_proj",
            )
            require_linear(
                down_proj,
                (decoder_hidden, intermediate_size),
                f"layer {layer_index} down_proj",
            )
            require_norm(
                attention_norm_module,
                decoder_hidden,
                f"layer {layer_index} input_layernorm",
            )
            require_norm(
                mlp_norm_module,
                decoder_hidden,
                f"layer {layer_index} post_attention_layernorm",
            )
            if (
                int(getattr(attention_module, "head_dim", -1)) != head_dim
                or int(getattr(attention_module, "num_key_value_groups", -1))
                != query_heads // key_value_heads
                or getattr(attention_module, "is_causal", None) is not True
                or getattr(attention_module, "sliding_window", None) is not None
                or getattr(layer, "attention_type", "full_attention")
                != "full_attention"
            ):
                raise DynamicModelInspectionError(
                    f"NEO Qwen3 layer {layer_index} attention metadata differs "
                    "from its full-attention grouped-query config."
                )
            expected_attention_parameters = {
                "q_proj.weight",
                "q_proj_hw.weight",
                "k_proj.weight",
                "k_proj_hw.weight",
                "v_proj.weight",
                "o_proj.weight",
                "q_norm.weight",
                "q_norm_h.weight",
                "q_norm_w.weight",
                "k_norm.weight",
                "k_norm_h.weight",
                "k_norm_w.weight",
            }
            if {
                name
                for name, _parameter in attention_module.named_parameters(
                    recurse=True,
                    remove_duplicate=False,
                )
            } != expected_attention_parameters:
                raise DynamicModelInspectionError(
                    f"NEO Qwen3 layer {layer_index} attention owns unsupported "
                    "learned tensors."
                )
            if {
                name
                for name, _parameter in mlp_module.named_parameters(
                    recurse=True,
                    remove_duplicate=False,
                )
            } != {
                "gate_proj.weight",
                "up_proj.weight",
                "down_proj.weight",
            }:
                raise DynamicModelInspectionError(
                    f"NEO Qwen3 layer {layer_index} MLP owns unsupported learned "
                    "tensors."
                )

            layer_path = _module_path(model, layer)
            attention_norm = builder.add_node(
                f"decoder.layer.{layer_index:03d}.attention_norm",
                CircuitKind.NORM,
                _module_path(model, attention_norm_module),
                layer_index=layer_index,
                inputs=(
                    _port(
                        "residual",
                        PortKind.RESIDUAL,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                ),
                outputs=(
                    _port(
                        "normalized",
                        PortKind.NORMALIZED_RESIDUAL,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                ),
                role_prefix="decoder_attention_norm",
                attributes={
                    "norm_kind": "rms_norm",
                    "epsilon": float(llm_config.rms_norm_eps),
                },
            )
            attention = builder.add_node(
                f"decoder.layer.{layer_index:03d}.attention",
                CircuitKind.ATTENTION,
                _module_path(model, attention_module),
                layer_index=layer_index,
                inputs=(
                    _port(
                        "residual",
                        PortKind.NORMALIZED_RESIDUAL,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                    _port(
                        "indexes",
                        PortKind.POSITION_IDS,
                        3,
                        "sequence",
                    ),
                    _port(
                        "mask",
                        PortKind.ATTENTION_MASK,
                        "batch",
                        "sequence",
                        "sequence",
                    ),
                    _port(
                        "cache_in",
                        PortKind.CACHE,
                        "layers",
                        stateful=True,
                    ),
                ),
                outputs=(
                    _port(
                        "delta",
                        PortKind.BRANCH_DELTA,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                    _port(
                        "cache_out",
                        PortKind.CACHE,
                        "layers",
                        stateful=True,
                    ),
                ),
                role_prefix="decoder_attention",
                attributes={
                    "projection_style": "split_qkv",
                    "query_heads": query_heads,
                    "key_value_heads": key_value_heads,
                    "head_dim": head_dim,
                    "grouped_query_attention": True,
                    "layer_type": "full_attention",
                    "temporal_query_projection": "q_proj",
                    "spatial_query_projection": "q_proj_hw",
                    "temporal_key_projection": "k_proj",
                    "spatial_key_projection": "k_proj_hw",
                    "position_axes": 3,
                },
            )
            attention_add = builder.add_node(
                f"decoder.layer.{layer_index:03d}.attention_residual",
                CircuitKind.RESIDUAL_ADD,
                layer_path,
                layer_index=layer_index,
                inputs=(
                    _port(
                        "base",
                        PortKind.RESIDUAL,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                    _port(
                        "delta",
                        PortKind.BRANCH_DELTA,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                ),
                outputs=(
                    _port(
                        "residual",
                        PortKind.RESIDUAL,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                ),
                attributes={"ordered_branch": "attention"},
            )
            mlp_norm = builder.add_node(
                f"decoder.layer.{layer_index:03d}.mlp_norm",
                CircuitKind.NORM,
                _module_path(model, mlp_norm_module),
                layer_index=layer_index,
                inputs=(
                    _port(
                        "residual",
                        PortKind.RESIDUAL,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                ),
                outputs=(
                    _port(
                        "normalized",
                        PortKind.NORMALIZED_RESIDUAL,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                ),
                role_prefix="decoder_mlp_norm",
                attributes={
                    "norm_kind": "rms_norm",
                    "epsilon": float(llm_config.rms_norm_eps),
                },
            )
            mlp = builder.add_node(
                f"decoder.layer.{layer_index:03d}.mlp",
                CircuitKind.GATED_MLP,
                _module_path(model, mlp_module),
                layer_index=layer_index,
                inputs=(
                    _port(
                        "residual",
                        PortKind.NORMALIZED_RESIDUAL,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                ),
                outputs=(
                    _port(
                        "delta",
                        PortKind.BRANCH_DELTA,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                ),
                role_prefix="decoder_mlp",
                attributes={
                    "intermediate_size": intermediate_size,
                    "activation": "silu",
                    "projection_layout": "output_input",
                },
            )
            mlp_add = builder.add_node(
                f"decoder.layer.{layer_index:03d}.mlp_residual",
                CircuitKind.RESIDUAL_ADD,
                layer_path,
                layer_index=layer_index,
                inputs=(
                    _port(
                        "base",
                        PortKind.RESIDUAL,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                    _port(
                        "delta",
                        PortKind.BRANCH_DELTA,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                ),
                outputs=(
                    _port(
                        "residual",
                        PortKind.RESIDUAL,
                        "batch",
                        "sequence",
                        decoder_hidden,
                    ),
                ),
                attributes={"ordered_branch": "mlp"},
            )
            builder.edge(
                current,
                current_port,
                attention_norm,
                "residual",
                EdgeKind.DATA,
            )
            builder.edge(
                attention_norm,
                "normalized",
                attention,
                "residual",
                EdgeKind.DATA,
            )
            builder.edge(
                position,
                "indexes",
                attention,
                "indexes",
                EdgeKind.CONTROL,
            )
            builder.edge(
                attention,
                "cache_out",
                attention,
                "cache_in",
                EdgeKind.CACHE,
                recurrent=True,
            )
            builder.edge(
                current,
                current_port,
                attention_add,
                "base",
                EdgeKind.RESIDUAL,
            )
            builder.edge(
                attention,
                "delta",
                attention_add,
                "delta",
                EdgeKind.RESIDUAL,
            )
            builder.edge(
                attention_add,
                "residual",
                mlp_norm,
                "residual",
                EdgeKind.DATA,
            )
            builder.edge(
                mlp_norm,
                "normalized",
                mlp,
                "residual",
                EdgeKind.DATA,
            )
            builder.edge(
                attention_add,
                "residual",
                mlp_add,
                "base",
                EdgeKind.RESIDUAL,
            )
            builder.edge(
                mlp,
                "delta",
                mlp_add,
                "delta",
                EdgeKind.RESIDUAL,
            )
            current = mlp_add
            current_port = "residual"

        require_norm(final_norm_module, decoder_hidden, "final norm")
        final_norm = builder.add_node(
            "decoder.output.final_norm",
            CircuitKind.NORM,
            final_norm_path,
            layer_index=None,
            inputs=(
                _port(
                    "residual",
                    PortKind.RESIDUAL,
                    "batch",
                    "sequence",
                    decoder_hidden,
                ),
            ),
            outputs=(
                _port(
                    "normalized",
                    PortKind.NORMALIZED_RESIDUAL,
                    "batch",
                    "sequence",
                    decoder_hidden,
                ),
            ),
            role_prefix="decoder_final_norm",
            attributes={
                "norm_kind": "rms_norm",
                "epsilon": float(llm_config.rms_norm_eps),
            },
        )
        head = builder.add_node(
            "decoder.output.lm_head",
            CircuitKind.OUTPUT_HEAD,
            lm_head_path,
            layer_index=None,
            inputs=(
                _port(
                    "residual",
                    PortKind.NORMALIZED_RESIDUAL,
                    "batch",
                    "sequence",
                    decoder_hidden,
                ),
            ),
            outputs=(
                _port(
                    "logits",
                    PortKind.LOGITS,
                    "batch",
                    "sequence",
                    vocab_size,
                    modality="text",
                ),
            ),
            role_prefix="decoder_output_head",
            attributes={
                "vocabulary_size": vocab_size,
                "tied_to_embedding": bool(llm_config.tie_word_embeddings),
            },
        )
        builder.edge(
            current,
            current_port,
            final_norm,
            "residual",
            EdgeKind.DATA,
        )
        builder.edge(
            final_norm,
            "normalized",
            head,
            "residual",
            EdgeKind.DATA,
        )
        builder.interfaces.append(
            ModalityInterface(
                interface_id="text.logits.output",
                modality="text",
                direction="output",
                node_id=head.node_id,
                port_name="logits",
                representation="token_logits",
            )
        )

        vision_parameter_count = sum(
            int(parameter.numel())
            for path, parameter in model.named_parameters(
                recurse=True,
                remove_duplicate=False,
            )
            if path.startswith(f"{vision_path}.")
        )
        return builder.finish(
            metadata={
                "inspector": self.name,
                "decoder_family": "qwen3",
                "layer_count": expected_layers,
                "hidden_size": decoder_hidden,
                "decoder_intermediate_size": int(llm_config.intermediate_size),
                "vision_hidden_size": vision_hidden,
                "vision_parameter_count": vision_parameter_count,
                "native_rgb_patch_path": True,
                "image_token_replacement_fusion": True,
                "native_temporal_height_width_attention": True,
            }
        )


class HFCausalDecoderInspector:
    """Compile explicitly validated GPT-2/Llama/Qwen/Gemma layouts."""

    name = "hf-causal-decoder"
    priority = 100
    _MODEL_TYPES = {"gpt2", "llama", "qwen3", "gemma2", "gemma3_text"}

    def supports(self, model: Any) -> bool:
        return getattr(getattr(model, "config", None), "model_type", None) in (
            self._MODEL_TYPES
        )

    def _build_layout(self, model: Any) -> Any:
        from .hf_arch_layout import build_decoder_layout

        return build_decoder_layout(model)

    def inspect(
        self,
        model: Any,
        *,
        model_name: str,
        tokenizer_fingerprint: str | None = None,
    ) -> ModelCircuitGraph:
        try:
            layout = self._build_layout(model)
        except Exception as exc:
            raise DynamicModelInspectionError(
                "The Hugging Face decoder failed its complete topology contract."
            ) from exc
        builder = _GraphBuilder(
            model=model,
            model_name=model_name,
            family=layout.family,
            tokenizer_fingerprint=tokenizer_fingerprint,
            groups=_storage_groups(model),
            nodes=[],
            edges=[],
            interfaces=[],
            captured=set(),
        )
        hidden = layout.hidden_size
        embedding = builder.add_node(
            "input.token_embedding",
            CircuitKind.TOKEN_EMBEDDING,
            layout.embeddings.token_path,
            layer_index=None,
            inputs=(_port("token_ids", PortKind.TOKEN_IDS, "batch", "sequence"),),
            outputs=(
                _port(
                    "residual",
                    PortKind.RESIDUAL,
                    "batch",
                    "sequence",
                    hidden,
                    modality="text",
                ),
            ),
            role_prefix="embedding",
            attributes={
                "vocabulary_size": layout.embeddings.vocab_size,
                "hidden_size": hidden,
                "token_scale": layout.embeddings.token_scale,
            },
        )
        builder.interfaces.append(
            ModalityInterface(
                interface_id="text.tokens.input",
                modality="text",
                direction="input",
                node_id=embedding.node_id,
                port_name="token_ids",
                representation="token_ids",
            )
        )
        position_path = (
            layout.embeddings.position_path
            if layout.embeddings.position_path is not None
            else layout.position.operator_paths[0]
        )
        position = builder.add_node(
            "input.position",
            CircuitKind.POSITION_ENCODING,
            position_path,
            layer_index=None,
            inputs=(_port("position_ids", PortKind.POSITION_IDS, "batch", "sequence"),),
            outputs=(_port("position", PortKind.GENERIC, "batch", "sequence", hidden),),
            role_prefix=(
                "position" if layout.embeddings.position_path is not None else None
            ),
            attributes={
                "kind": layout.position.kind,
                "max_positions": layout.position.max_positions,
                "rotary_dim": layout.position.rotary_dim,
                "rope_theta": layout.position.rope_theta,
                "rope_type": layout.position.rope_type,
            },
        )
        current = embedding
        current_port = "residual"
        if layout.embeddings.position_path is not None:
            merge = builder.add_node(
                "input.embedding_position_add",
                CircuitKind.RESIDUAL_ADD,
                layout.embeddings.token_path,
                layer_index=None,
                inputs=(
                    _port("base", PortKind.RESIDUAL, "batch", "sequence", hidden),
                    _port("delta", PortKind.BRANCH_DELTA, "batch", "sequence", hidden),
                ),
                outputs=(
                    _port("residual", PortKind.RESIDUAL, "batch", "sequence", hidden),
                ),
            )
            builder.edge(current, current_port, merge, "base", EdgeKind.RESIDUAL)
            builder.edge(position, "position", merge, "delta", EdgeKind.CONTROL)
            current = merge
            current_port = "residual"

        for block in layout.blocks:
            prefix = f"layer.{block.index:03d}"
            attention_norm = builder.add_node(
                f"{prefix}.attention_norm",
                CircuitKind.NORM,
                block.attention_input_norm.path,
                layer_index=block.index,
                inputs=(
                    _port("residual", PortKind.RESIDUAL, "batch", "sequence", hidden),
                ),
                outputs=(
                    _port(
                        "normalized",
                        PortKind.NORMALIZED_RESIDUAL,
                        "batch",
                        "sequence",
                        hidden,
                    ),
                ),
                role_prefix="attention_norm",
                attributes={
                    "norm_kind": block.attention_input_norm.kind,
                    "epsilon": block.attention_input_norm.epsilon,
                    "weight_kind": block.attention_input_norm.weight_kind,
                },
            )
            attention = builder.add_node(
                f"{prefix}.attention",
                CircuitKind.ATTENTION,
                block.attention.module_path,
                layer_index=block.index,
                inputs=(
                    _port(
                        "residual",
                        PortKind.NORMALIZED_RESIDUAL,
                        "batch",
                        "sequence",
                        hidden,
                    ),
                    _port("position", PortKind.GENERIC, "batch", "sequence", hidden),
                    _port(
                        "mask",
                        PortKind.ATTENTION_MASK,
                        "batch",
                        "sequence",
                        "sequence",
                    ),
                    _port("cache_in", PortKind.CACHE, "layers", stateful=True),
                ),
                outputs=(
                    _port(
                        "delta",
                        PortKind.BRANCH_DELTA,
                        "batch",
                        "sequence",
                        hidden,
                    ),
                    _port("cache_out", PortKind.CACHE, "layers", stateful=True),
                ),
                role_prefix="attention",
                attributes={
                    "projection_style": block.attention.projection_style,
                    "query_heads": block.attention.query_heads,
                    "key_value_heads": block.attention.key_value_heads,
                    "head_dim": block.attention.head_dim,
                    "grouped_query_attention": (
                        block.attention.grouped_query_attention
                    ),
                    "layer_type": block.attention.layer_type,
                    "sliding_window": block.attention.sliding_window,
                    "residual_source": block.residual.attention_residual_source,
                },
            )
            builder.edge(
                current, current_port, attention_norm, "residual", EdgeKind.DATA
            )
            builder.edge(
                attention_norm, "normalized", attention, "residual", EdgeKind.DATA
            )
            builder.edge(position, "position", attention, "position", EdgeKind.CONTROL)
            builder.edge(
                attention,
                "cache_out",
                attention,
                "cache_in",
                EdgeKind.CACHE,
                recurrent=True,
            )
            branch = attention
            branch_port = "delta"
            if block.attention_output_norm is not None:
                post_attention_norm = builder.add_node(
                    f"{prefix}.attention_output_norm",
                    CircuitKind.NORM,
                    block.attention_output_norm.path,
                    layer_index=block.index,
                    inputs=(
                        _port(
                            "residual",
                            PortKind.BRANCH_DELTA,
                            "batch",
                            "sequence",
                            hidden,
                        ),
                    ),
                    outputs=(
                        _port(
                            "normalized",
                            PortKind.BRANCH_DELTA,
                            "batch",
                            "sequence",
                            hidden,
                        ),
                    ),
                    role_prefix="attention_output_norm",
                    attributes={
                        "norm_kind": block.attention_output_norm.kind,
                        "epsilon": block.attention_output_norm.epsilon,
                    },
                )
                builder.edge(
                    attention,
                    "delta",
                    post_attention_norm,
                    "residual",
                    EdgeKind.DATA,
                )
                branch = post_attention_norm
                branch_port = "normalized"
            attention_add = builder.add_node(
                f"{prefix}.attention_residual",
                CircuitKind.RESIDUAL_ADD,
                block.path,
                layer_index=block.index,
                inputs=(
                    _port("base", PortKind.RESIDUAL, "batch", "sequence", hidden),
                    _port("delta", PortKind.BRANCH_DELTA, "batch", "sequence", hidden),
                ),
                outputs=(
                    _port("residual", PortKind.RESIDUAL, "batch", "sequence", hidden),
                ),
                attributes={"ordered_branch": "attention"},
            )
            builder.edge(
                current, current_port, attention_add, "base", EdgeKind.RESIDUAL
            )
            builder.edge(branch, branch_port, attention_add, "delta", EdgeKind.RESIDUAL)

            mlp_norm = builder.add_node(
                f"{prefix}.mlp_norm",
                CircuitKind.NORM,
                block.mlp_input_norm.path,
                layer_index=block.index,
                inputs=(
                    _port("residual", PortKind.RESIDUAL, "batch", "sequence", hidden),
                ),
                outputs=(
                    _port(
                        "normalized",
                        PortKind.NORMALIZED_RESIDUAL,
                        "batch",
                        "sequence",
                        hidden,
                    ),
                ),
                role_prefix="mlp_norm",
                attributes={
                    "norm_kind": block.mlp_input_norm.kind,
                    "epsilon": block.mlp_input_norm.epsilon,
                    "weight_kind": block.mlp_input_norm.weight_kind,
                },
            )
            mlp_kind = (
                CircuitKind.GATED_MLP
                if block.mlp.kind == "gated"
                else CircuitKind.DENSE_MLP
            )
            mlp = builder.add_node(
                f"{prefix}.mlp",
                mlp_kind,
                block.mlp.module_path,
                layer_index=block.index,
                inputs=(
                    _port(
                        "residual",
                        PortKind.NORMALIZED_RESIDUAL,
                        "batch",
                        "sequence",
                        hidden,
                    ),
                ),
                outputs=(
                    _port(
                        "delta",
                        PortKind.BRANCH_DELTA,
                        "batch",
                        "sequence",
                        hidden,
                    ),
                ),
                role_prefix="mlp",
                attributes={
                    "intermediate_size": block.mlp.intermediate_size,
                    "activation": block.mlp.activation_name,
                    "projection_layout": block.mlp.projection_weight_layout,
                    "residual_source": block.residual.mlp_residual_source,
                },
            )
            builder.edge(
                attention_add,
                "residual",
                mlp_norm,
                "residual",
                EdgeKind.DATA,
            )
            builder.edge(mlp_norm, "normalized", mlp, "residual", EdgeKind.DATA)
            mlp_branch = mlp
            mlp_branch_port = "delta"
            if block.mlp_output_norm is not None:
                post_mlp_norm = builder.add_node(
                    f"{prefix}.mlp_output_norm",
                    CircuitKind.NORM,
                    block.mlp_output_norm.path,
                    layer_index=block.index,
                    inputs=(
                        _port(
                            "residual",
                            PortKind.BRANCH_DELTA,
                            "batch",
                            "sequence",
                            hidden,
                        ),
                    ),
                    outputs=(
                        _port(
                            "normalized",
                            PortKind.BRANCH_DELTA,
                            "batch",
                            "sequence",
                            hidden,
                        ),
                    ),
                    role_prefix="mlp_output_norm",
                    attributes={
                        "norm_kind": block.mlp_output_norm.kind,
                        "epsilon": block.mlp_output_norm.epsilon,
                    },
                )
                builder.edge(mlp, "delta", post_mlp_norm, "residual", EdgeKind.DATA)
                mlp_branch = post_mlp_norm
                mlp_branch_port = "normalized"
            mlp_add = builder.add_node(
                f"{prefix}.mlp_residual",
                CircuitKind.RESIDUAL_ADD,
                block.path,
                layer_index=block.index,
                inputs=(
                    _port("base", PortKind.RESIDUAL, "batch", "sequence", hidden),
                    _port("delta", PortKind.BRANCH_DELTA, "batch", "sequence", hidden),
                ),
                outputs=(
                    _port("residual", PortKind.RESIDUAL, "batch", "sequence", hidden),
                ),
                attributes={"ordered_branch": "mlp"},
            )
            builder.edge(
                attention_add,
                "residual",
                mlp_add,
                "base",
                EdgeKind.RESIDUAL,
            )
            builder.edge(
                mlp_branch,
                mlp_branch_port,
                mlp_add,
                "delta",
                EdgeKind.RESIDUAL,
            )
            current = mlp_add
            current_port = "residual"

        final_norm = builder.add_node(
            "output.final_norm",
            CircuitKind.NORM,
            layout.final_norm.path,
            layer_index=None,
            inputs=(_port("residual", PortKind.RESIDUAL, "batch", "sequence", hidden),),
            outputs=(
                _port(
                    "normalized",
                    PortKind.NORMALIZED_RESIDUAL,
                    "batch",
                    "sequence",
                    hidden,
                ),
            ),
            role_prefix="final_norm",
            attributes={
                "norm_kind": layout.final_norm.kind,
                "epsilon": layout.final_norm.epsilon,
                "weight_kind": layout.final_norm.weight_kind,
            },
        )
        head = builder.add_node(
            "output.lm_head",
            CircuitKind.OUTPUT_HEAD,
            layout.lm_head_path,
            layer_index=None,
            inputs=(
                _port(
                    "residual",
                    PortKind.NORMALIZED_RESIDUAL,
                    "batch",
                    "sequence",
                    hidden,
                ),
            ),
            outputs=(
                _port(
                    "logits",
                    PortKind.LOGITS,
                    "batch",
                    "sequence",
                    layout.embeddings.vocab_size,
                    modality="text",
                ),
            ),
            role_prefix="output_head",
            attributes={
                "vocabulary_size": layout.embeddings.vocab_size,
                "tied_to_embedding": layout.tie_word_embeddings,
                "logit_softcap": layout.final_logit_softcap,
            },
        )
        builder.edge(current, current_port, final_norm, "residual", EdgeKind.DATA)
        builder.edge(final_norm, "normalized", head, "residual", EdgeKind.DATA)
        builder.interfaces.append(
            ModalityInterface(
                interface_id="text.logits.output",
                modality="text",
                direction="output",
                node_id=head.node_id,
                port_name="logits",
                representation="token_logits",
            )
        )
        return builder.finish(
            metadata={
                "inspector": self.name,
                "position_policy": layout.position.kind,
                "layer_count": layout.layer_count,
                "hidden_size": hidden,
                "tie_word_embeddings": layout.tie_word_embeddings,
            }
        )


class Qwen3CompensatedCircuitInspector(HFCausalDecoderInspector):
    """Inspect a Qwen3 decoder with authenticated per-layer MLP widths."""

    name = "qwen3-compensated-ragged"
    priority = 250

    def supports(self, model: Any) -> bool:
        from .hf_qwen_compensated_runtime import (
            LAYOUT_ATTRIBUTE,
            REPORT_ATTRIBUTE,
        )

        return (
            getattr(getattr(model, "config", None), "model_type", None) == "qwen3"
            and hasattr(model, LAYOUT_ATTRIBUTE)
            and hasattr(model, REPORT_ATTRIBUTE)
        )

    def _build_layout(self, model: Any) -> Any:
        from .hf_qwen_compensated_runtime import (
            LAYOUT_ATTRIBUTE,
            validate_compensated_qwen3_runtime,
        )

        validate_compensated_qwen3_runtime(model)
        return getattr(model, LAYOUT_ATTRIBUTE)

    def inspect(
        self,
        model: Any,
        *,
        model_name: str,
        tokenizer_fingerprint: str | None = None,
    ) -> ModelCircuitGraph:
        graph = super().inspect(
            model,
            model_name=model_name,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
        report = validate_compensated_qwen3_runtime_for_metadata(model)
        metadata = dict(graph.metadata)
        metadata.update(
            {
                "inspector": self.name,
                "compensated_ragged": True,
                "configured_intermediate_size": int(
                    report["configured_intermediate_size"]
                ),
                "compact_layer_indices": ",".join(
                    str(value) for value in report["compact_layer_indices"]
                ),
                "layer_widths_sha256": canonical_sha256(report["layer_widths"]),
            }
        )
        return ModelCircuitGraph(
            model_name=graph.model_name,
            family=graph.family,
            model_class=graph.model_class,
            nodes=graph.nodes,
            edges=graph.edges,
            interfaces=graph.interfaces,
            logical_parameter_count=graph.logical_parameter_count,
            unique_parameter_count=graph.unique_parameter_count,
            unique_parameter_bytes=graph.unique_parameter_bytes,
            config_fingerprint=graph.config_fingerprint,
            tokenizer_fingerprint=graph.tokenizer_fingerprint,
            metadata=tuple(metadata.items()),
        )


def validate_compensated_qwen3_runtime_for_metadata(
    model: Any,
) -> Mapping[str, Any]:
    """Late import boundary used by the compensated inspector."""

    from .hf_qwen_compensated_runtime import validate_compensated_qwen3_runtime

    return validate_compensated_qwen3_runtime(model)


_LFM2_HYBRID_MODEL_METHOD = "lfm2_hybrid_residual_attention_model"
_LFM2_PRODUCER_FIRST_MODEL_METHOD = (
    "lfm2_producer_first_residual_attention_model"
)
_LFM2_DENSE_BUDGET_MODEL_METHOD = (
    "lfm2_dense_budget_residual_attention_model"
)
_LFM2_DYNAMIC_CONFIG_MARKERS = (
    "neural_forge_dynamic_blank_config_schema",
    "neural_forge_dynamic_physical_layout",
    "neural_forge_dynamic_physical_layout_sha256",
)
_LFM2_DYNAMIC_RUNTIME_MARKER = "_neural_forge_dynamic_blank_runtime_sha256"


def _authenticated_lfm2_heterogeneous_blank(
    model: Any,
) -> tuple[Any, Mapping[str, Any]] | None:
    """Return one live sealed heterogeneous blank or fail closed.

    Detection is config-driven.  Merely resembling the dynamic module shapes
    is insufficient: the canonical descriptor, model-local runtime seal, and
    the materializer's complete live physical audit must all agree.  The
    runtime marker check occurs before the idempotent materializer call so an
    inspector can never materialize a fresh native object as a side effect.
    """

    config = getattr(model, "config", None)
    if config is None:
        return None
    marker_present = any(
        hasattr(config, name) for name in _LFM2_DYNAMIC_CONFIG_MARKERS
    ) or hasattr(model, _LFM2_DYNAMIC_RUNTIME_MARKER)
    if not marker_present:
        return None

    from .hf_lfm2_dynamic_blank_materializer import (
        LFM2_DYNAMIC_BLANK_MATERIALIZATION_SCHEMA,
        Lfm2DynamicBlankMaterializationError,
        extract_lfm2_dynamic_physical_layout_from_config,
        materialize_lfm2_dynamic_blank_child,
    )
    from .hf_lfm2_dynamic_physical_layout import (
        Lfm2DynamicPhysicalLayoutError,
        Lfm2HeterogeneousPhysicalLayoutDescriptor,
        lfm2_expected_heterogeneous_physical_state_shapes,
    )

    try:
        layout = extract_lfm2_dynamic_physical_layout_from_config(config)
    except (Lfm2DynamicBlankMaterializationError, Lfm2DynamicPhysicalLayoutError) as exc:
        raise DynamicModelInspectionError(
            "LFM2 dynamic physical-layout config authentication failed."
        ) from exc
    if type(layout) is not Lfm2HeterogeneousPhysicalLayoutDescriptor:
        # The original v2 dynamic layout is outside this heterogeneous path.
        # Its existing native fail-closed behavior remains unchanged.
        return None
    if (
        getattr(model, _LFM2_DYNAMIC_RUNTIME_MARKER, None)
        != layout.descriptor_sha256
    ):
        raise DynamicModelInspectionError(
            "Sealed heterogeneous LFM2 config is not an already-materialized runtime."
        )
    try:
        receipt = materialize_lfm2_dynamic_blank_child(
            model,
            config,
            layout,
            expected_descriptor_sha256=layout.descriptor_sha256,
        )
    except (Lfm2DynamicBlankMaterializationError, Lfm2DynamicPhysicalLayoutError) as exc:
        raise DynamicModelInspectionError(
            "Heterogeneous LFM2 live physical materialization authentication failed."
        ) from exc
    expected_shapes = lfm2_expected_heterogeneous_physical_state_shapes(layout)
    audit = receipt.get("physical_materialization_audit")
    expected_elements = layout.accounting.unique_parameter_elements
    if (
        receipt.get("schema") != LFM2_DYNAMIC_BLANK_MATERIALIZATION_SCHEMA
        or receipt.get("descriptor_sha256") != layout.descriptor_sha256
        or receipt.get("parameter_state_key_count") != len(expected_shapes)
        or receipt.get("parameter_elements") != expected_elements
        or not isinstance(audit, Mapping)
        or audit.get("descriptor_sha256") != layout.descriptor_sha256
        or audit.get("parameter_elements") != expected_elements
        or audit.get("parameter_bytes_fp32") != 4 * expected_elements
        or audit.get("runtime_parameter_alias_group_count") != 0
        or audit.get("cross_layer_parameter_sharing") is not False
        or audit.get("parameters_quantized") != 0
        or audit.get("sparse_parameter_storage") is not False
        or audit.get("pruning_masks_present") is not False
        or audit.get("runtime_bridge_parameter_elements") != 0
        or audit.get("runtime_sidecar_parameter_elements") != 0
    ):
        raise DynamicModelInspectionError(
            "Heterogeneous LFM2 materialization receipt violates its physical audit."
        )
    return layout, receipt


def _validated_lfm2_hybrid_markers(
    config: Any,
    *,
    layer_count: int,
) -> dict[str, Any]:
    """Accept only the compiler-owned, structurally complete marker bundle."""

    names = (
        "_neural_forge_fold_method_id",
        "_neural_forge_ordered_source_spans",
        "_neural_forge_target_layer_types",
    )
    values = {name: getattr(config, name, None) for name in names}
    present = {name for name, value in values.items() if value is not None}
    dense_base_names = (
        "_neural_forge_dense_topology_catalog",
        "_neural_forge_dense_ordered_source_spans",
        "_neural_forge_dense_target_layer_types",
        "_neural_forge_tokenizer_bridge_required",
    )
    dense_names = (*dense_base_names, *sorted(TOKENIZER_BRIDGE_CONFIG_KEYS))
    dense_values = {
        name: getattr(config, name, None) for name in dense_names
    }
    dense_present = {
        name for name, value in dense_values.items() if value is not None
    }
    if present and dense_present:
        raise DynamicModelInspectionError(
            "LFM2 legacy and dense-budget fold marker bundles cannot coexist."
        )
    if dense_present:
        if not set(dense_base_names).issubset(dense_present):
            raise DynamicModelInspectionError(
                "Dense-budget LFM2 markers must be present as one complete bundle."
            )
        bridge_required = dense_values[
            "_neural_forge_tokenizer_bridge_required"
        ]
        if not isinstance(bridge_required, bool):
            raise DynamicModelInspectionError(
                "Dense-budget tokenizer bridge marker must be boolean."
            )
        expected_dense_names = set(dense_base_names)
        if bridge_required:
            expected_dense_names.update(TOKENIZER_BRIDGE_CONFIG_KEYS)
        if dense_present != expected_dense_names:
            raise DynamicModelInspectionError(
                "Dense-budget tokenizer bridge seals must be present exactly "
                "when tokenizer bridging is required."
            )
        catalog_id = dense_values["_neural_forge_dense_topology_catalog"]
        if not isinstance(catalog_id, str) or not catalog_id.strip():
            raise DynamicModelInspectionError(
                "Dense-budget LFM2 catalog marker must be nonempty."
            )
        bridge_metadata: dict[str, str] = {}
        if bridge_required:
            for name in TOKENIZER_BRIDGE_CONFIG_KEYS:
                value = dense_values[name]
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in value
                    )
                ):
                    raise DynamicModelInspectionError(
                        f"Dense-budget tokenizer bridge seal {name!r} is invalid."
                    )
                bridge_metadata[name] = value
        layer_types_value = dense_values[
            "_neural_forge_dense_target_layer_types"
        ]
        if not isinstance(layer_types_value, str):
            raise DynamicModelInspectionError(
                "Dense-budget target-layer marker must be a string."
            )
        layer_types = tuple(
            part.strip() for part in layer_types_value.split(",")
        )
        if (
            len(layer_types) != layer_count
            or any(
                value not in {"conv", "full_attention"}
                for value in layer_types
            )
            or tuple(
                str(value) for value in getattr(config, "layer_types", ())
            )
            != layer_types
        ):
            raise DynamicModelInspectionError(
                "Dense-budget target-layer marker differs from config topology."
            )
        spans_value = dense_values[
            "_neural_forge_dense_ordered_source_spans"
        ]
        if not isinstance(spans_value, str) or not spans_value:
            raise DynamicModelInspectionError(
                "Dense-budget ordered-source-span marker must be nonempty."
            )
        try:
            spans = tuple(
                tuple(int(part.strip()) for part in span.split(","))
                for span in spans_value.split(";")
            )
        except ValueError as exc:
            raise DynamicModelInspectionError(
                "Dense-budget ordered-source spans contain a non-integer."
            ) from exc
        flattened = tuple(index for span in spans for index in span)
        if (
            len(spans) != layer_count
            or any(not span for span in spans)
            or any(
                tuple(range(span[0], span[-1] + 1)) != span
                or any(index < 0 for index in span)
                for span in spans
            )
            or not flattened
            or flattened != tuple(range(flattened[-1] + 1))
        ):
            raise DynamicModelInspectionError(
                "Dense-budget source spans must provide complete ordered "
                "contiguous source coverage."
            )
        result = {
            "fold_method_id": _LFM2_DENSE_BUDGET_MODEL_METHOD,
            "dense_topology_catalog": catalog_id,
            "ordered_source_spans": spans_value,
            "target_layer_types": layer_types_value,
            "tokenizer_bridge_required": bridge_required,
        }
        if bridge_required:
            result.update(
                {
                    "tokenizer_bridge_artifact_sha256": bridge_metadata[
                        TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY
                    ],
                    "tokenizer_transport_profile_sha256": bridge_metadata[
                        TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY
                    ],
                    "parent_tokenizer_semantics_sha256": bridge_metadata[
                        PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY
                    ],
                    "child_tokenizer_semantics_sha256": bridge_metadata[
                        CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY
                    ],
                }
            )
        return result
    if not present:
        return {}
    if values["_neural_forge_fold_method_id"] == (
        _LFM2_PRODUCER_FIRST_MODEL_METHOD
    ):
        if present != {"_neural_forge_fold_method_id"}:
            raise DynamicModelInspectionError(
                "Full-depth producer-first LFM2 markers must not include "
                "depth spans or target-layer overrides."
            )
        return {"fold_method_id": _LFM2_PRODUCER_FIRST_MODEL_METHOD}
    if present != set(names):
        raise DynamicModelInspectionError(
            "LFM2 compiler fold markers must be present as one complete bundle."
        )
    if values["_neural_forge_fold_method_id"] != _LFM2_HYBRID_MODEL_METHOD:
        raise DynamicModelInspectionError(
            "LFM2 compiler fold method marker is unknown."
        )
    layer_types_value = values["_neural_forge_target_layer_types"]
    if not isinstance(layer_types_value, str):
        raise DynamicModelInspectionError(
            "LFM2 compiler target-layer marker must be a string."
        )
    layer_types = tuple(part.strip() for part in layer_types_value.split(","))
    if (
        len(layer_types) != layer_count
        or any(value not in {"conv", "full_attention"} for value in layer_types)
    ):
        raise DynamicModelInspectionError(
            "LFM2 compiler target-layer marker differs from target depth."
        )
    configured_types = tuple(str(value) for value in getattr(config, "layer_types", ()))
    if configured_types != layer_types:
        raise DynamicModelInspectionError(
            "LFM2 compiler target-layer marker differs from config.layer_types."
        )
    spans_value = values["_neural_forge_ordered_source_spans"]
    if not isinstance(spans_value, str) or not spans_value:
        raise DynamicModelInspectionError(
            "LFM2 compiler ordered-source-span marker must be a string."
        )
    try:
        spans = tuple(
            tuple(int(part.strip()) for part in span.split(","))
            for span in spans_value.split(";")
        )
    except ValueError as exc:
        raise DynamicModelInspectionError(
            "LFM2 compiler ordered-source-span marker contains a non-integer."
        ) from exc
    if (
        len(spans) != layer_count
        or any(not span for span in spans)
        or any(
            tuple(range(span[0], span[-1] + 1)) != span
            or any(index < 0 for index in span)
            for span in spans
        )
    ):
        raise DynamicModelInspectionError(
            "LFM2 compiler ordered-source spans are not contiguous target spans."
        )
    flattened = tuple(index for span in spans for index in span)
    if (
        not flattened
        or flattened != tuple(range(flattened[-1] + 1))
        or len(flattened) <= layer_count
    ):
        raise DynamicModelInspectionError(
            "LFM2 compiler ordered-source spans do not encode strict depth "
            "compaction with complete ordered source coverage."
        )
    return {
        "fold_method_id": _LFM2_HYBRID_MODEL_METHOD,
        "ordered_source_spans": spans_value,
        "target_layer_types": layer_types_value,
    }


class Lfm2CircuitInspector:
    """Compile Liquid LFM2 attention/short-conv hybrid decoders."""

    name = "lfm2-hybrid-decoder"
    priority = 200

    def supports(self, model: Any) -> bool:
        return getattr(getattr(model, "config", None), "model_type", None) == "lfm2"

    def inspect(
        self,
        model: Any,
        *,
        model_name: str,
        tokenizer_fingerprint: str | None = None,
    ) -> ModelCircuitGraph:
        from .hf_lfm2_canonical import (
            Lfm2AttentionContract,
            Lfm2HeterogeneousLayerContract,
            Lfm2HeterogeneousShortConvContract,
            Lfm2ShortConvContract,
            inspect_lfm2_heterogeneous_layer,
            inspect_lfm2_layer,
        )

        heterogeneous_authentication = _authenticated_lfm2_heterogeneous_blank(
            model
        )
        heterogeneous_layout = (
            heterogeneous_authentication[0]
            if heterogeneous_authentication is not None
            else None
        )
        heterogeneous_receipt = (
            heterogeneous_authentication[1]
            if heterogeneous_authentication is not None
            else None
        )

        get_input = getattr(model, "get_input_embeddings", None)
        get_output = getattr(model, "get_output_embeddings", None)
        if not callable(get_input) or not callable(get_output):
            raise DynamicModelInspectionError(
                "LFM2 inspection requires causal-LM input/output embedding accessors."
            )
        embedding_module = get_input()
        head_module = get_output()
        if embedding_module is None or head_module is None:
            raise DynamicModelInspectionError(
                "LFM2 causal-LM input/output embeddings are unavailable."
            )
        core = getattr(model, "model", model)
        layers_value = getattr(core, "layers", None)
        try:
            layers = tuple(layers_value)
        except TypeError as exc:
            raise DynamicModelInspectionError(
                "Could not locate the LFM2 layer container."
            ) from exc
        if not layers:
            raise DynamicModelInspectionError("The LFM2 layer container is empty.")
        embedding_path = _module_path(model, embedding_module)
        head_path = _module_path(model, head_module)
        position_module = getattr(core, "rotary_emb", None)
        final_norm_module = getattr(core, "embedding_norm", None)
        if position_module is None or final_norm_module is None:
            raise DynamicModelInspectionError(
                "LFM2 rotary embedding or final norm is missing."
            )
        position_path = _module_path(model, position_module)
        final_norm_path = _module_path(model, final_norm_module)
        builder = _GraphBuilder(
            model=model,
            model_name=model_name,
            family="lfm2",
            tokenizer_fingerprint=tokenizer_fingerprint,
            groups=_storage_groups(model),
            nodes=[],
            edges=[],
            interfaces=[],
            captured=set(),
        )
        hidden = int(getattr(model.config, "hidden_size"))
        vocab = int(getattr(model.config, "vocab_size"))
        embedding = builder.add_node(
            "input.token_embedding",
            CircuitKind.TOKEN_EMBEDDING,
            embedding_path,
            layer_index=None,
            inputs=(_port("token_ids", PortKind.TOKEN_IDS, "batch", "sequence"),),
            outputs=(
                _port(
                    "residual",
                    PortKind.RESIDUAL,
                    "batch",
                    "sequence",
                    hidden,
                    modality="text",
                ),
            ),
            role_prefix="embedding",
            attributes={"vocabulary_size": vocab, "hidden_size": hidden},
        )
        position = builder.add_node(
            "input.position",
            CircuitKind.POSITION_ENCODING,
            position_path,
            layer_index=None,
            inputs=(_port("position_ids", PortKind.POSITION_IDS, "batch", "sequence"),),
            outputs=(_port("position", PortKind.GENERIC, "batch", "sequence", hidden),),
            attributes={"kind": "rope"},
        )
        builder.interfaces.append(
            ModalityInterface(
                interface_id="text.tokens.input",
                modality="text",
                direction="input",
                node_id=embedding.node_id,
                port_name="token_ids",
                representation="token_ids",
            )
        )
        current = embedding
        for index, layer in enumerate(layers):
            if heterogeneous_layout is None:
                contract = inspect_lfm2_layer(layer)
            else:
                contract = inspect_lfm2_heterogeneous_layer(
                    layer,
                    heterogeneous_layout,
                )
                if not isinstance(contract, Lfm2HeterogeneousLayerContract):
                    raise DynamicModelInspectionError(
                        "Heterogeneous LFM2 canonical contract type is inconsistent."
                    )
            if contract.layer_index != index:
                raise DynamicModelInspectionError(
                    "LFM2 layer order disagrees with canonical layer_index."
                )
            layer_path = _module_path(model, layer)
            prefix = f"layer.{index:03d}"
            operator_norm = builder.add_node(
                f"{prefix}.operator_norm",
                CircuitKind.NORM,
                f"{layer_path}.operator_norm",
                layer_index=index,
                inputs=(
                    _port("residual", PortKind.RESIDUAL, "batch", "sequence", hidden),
                ),
                outputs=(
                    _port(
                        "normalized",
                        PortKind.NORMALIZED_RESIDUAL,
                        "batch",
                        "sequence",
                        hidden,
                    ),
                ),
                role_prefix="operator_norm",
                attributes={"norm_kind": "rms_norm"},
            )
            if contract.kind == "full_attention":
                operator_contract = contract.operator
                if not isinstance(operator_contract, Lfm2AttentionContract):
                    raise DynamicModelInspectionError(
                        "LFM2 attention contract type is inconsistent."
                    )
                operator_module = layer.self_attn
                operator_kind = CircuitKind.ATTENTION
                state_kind = PortKind.CACHE
                edge_kind = EdgeKind.CACHE
                attributes = {
                    "query_heads": operator_contract.query_heads,
                    "key_value_heads": operator_contract.key_value_heads,
                    "head_dim": operator_contract.head_dim,
                    "grouped_query_attention": (
                        operator_contract.query_heads
                        != operator_contract.key_value_heads
                    ),
                    "qk_rmsnorm": operator_contract.qk_rmsnorm,
                    "operator_type": "full_attention",
                }
                if heterogeneous_layout is not None:
                    attributes.update(
                        {
                            "external_hidden_size": contract.hidden_size,
                            "query_projection_width": (
                                operator_contract.query_heads
                                * operator_contract.head_dim
                            ),
                            "key_value_projection_width": (
                                operator_contract.key_value_heads
                                * operator_contract.head_dim
                            ),
                            "selection_evidence_sha256": (
                                contract.selection_evidence_sha256
                            ),
                            "sealed_heterogeneous_physical_geometry": True,
                        }
                    )
            else:
                operator_contract = contract.operator
                if not isinstance(
                    operator_contract,
                    (
                        Lfm2ShortConvContract,
                        Lfm2HeterogeneousShortConvContract,
                    ),
                ):
                    raise DynamicModelInspectionError(
                        "LFM2 short-conv contract type is inconsistent."
                    )
                operator_module = layer.conv
                operator_kind = CircuitKind.SHORT_CONV
                state_kind = PortKind.RECURRENT_STATE
                edge_kind = EdgeKind.STATE
                attributes = {
                    "fir_length": operator_contract.fir_length,
                    "groups": operator_contract.groups,
                    "input_chunk_order": "b_c_x",
                    "operator_type": "short_conv",
                    "not_attention_qkv": True,
                }
                if isinstance(
                    operator_contract,
                    Lfm2HeterogeneousShortConvContract,
                ):
                    attributes.update(
                        {
                            "external_hidden_size": (
                                operator_contract.external_hidden_size
                            ),
                            "internal_channel_width": (
                                operator_contract.internal_channel_width
                            ),
                            "input_projection_rows": (
                                operator_contract.input_projection_shape[0]
                            ),
                            "writer_topology": operator_contract.writer_topology,
                            "writer_rank": operator_contract.writer_rank,
                            "writer_factor_count": len(
                                operator_contract.output_projection_shapes
                            ),
                            "selection_evidence_sha256": (
                                contract.selection_evidence_sha256
                            ),
                            "sealed_heterogeneous_physical_geometry": True,
                        }
                    )
            operator_path = _module_path(model, operator_module)
            operator = builder.add_node(
                f"{prefix}.operator",
                operator_kind,
                operator_path,
                layer_index=index,
                inputs=(
                    _port(
                        "residual",
                        PortKind.NORMALIZED_RESIDUAL,
                        "batch",
                        "sequence",
                        hidden,
                    ),
                    _port("position", PortKind.GENERIC, "batch", "sequence", hidden),
                    _port(
                        "mask",
                        PortKind.ATTENTION_MASK,
                        "batch",
                        "sequence",
                        "sequence",
                    ),
                    _port("state_in", state_kind, "layers", stateful=True),
                ),
                outputs=(
                    _port(
                        "delta",
                        PortKind.BRANCH_DELTA,
                        "batch",
                        "sequence",
                        hidden,
                    ),
                    _port("state_out", state_kind, "layers", stateful=True),
                ),
                role_prefix="operator",
                attributes=attributes,
            )
            operator_add = builder.add_node(
                f"{prefix}.operator_residual",
                CircuitKind.RESIDUAL_ADD,
                layer_path,
                layer_index=index,
                inputs=(
                    _port("base", PortKind.RESIDUAL, "batch", "sequence", hidden),
                    _port("delta", PortKind.BRANCH_DELTA, "batch", "sequence", hidden),
                ),
                outputs=(
                    _port("residual", PortKind.RESIDUAL, "batch", "sequence", hidden),
                ),
                attributes={"ordered_branch": "operator"},
            )
            builder.edge(current, "residual", operator_norm, "residual", EdgeKind.DATA)
            builder.edge(
                operator_norm, "normalized", operator, "residual", EdgeKind.DATA
            )
            builder.edge(position, "position", operator, "position", EdgeKind.CONTROL)
            builder.edge(
                operator,
                "state_out",
                operator,
                "state_in",
                edge_kind,
                recurrent=True,
            )
            builder.edge(current, "residual", operator_add, "base", EdgeKind.RESIDUAL)
            builder.edge(operator, "delta", operator_add, "delta", EdgeKind.RESIDUAL)

            ffn_norm = builder.add_node(
                f"{prefix}.mlp_norm",
                CircuitKind.NORM,
                f"{layer_path}.ffn_norm",
                layer_index=index,
                inputs=(
                    _port("residual", PortKind.RESIDUAL, "batch", "sequence", hidden),
                ),
                outputs=(
                    _port(
                        "normalized",
                        PortKind.NORMALIZED_RESIDUAL,
                        "batch",
                        "sequence",
                        hidden,
                    ),
                ),
                role_prefix="mlp_norm",
                attributes={"norm_kind": "rms_norm"},
            )
            mlp_path = f"{layer_path}.feed_forward"
            mlp_attributes: dict[str, Any] = {
                "intermediate_size": contract.ffn_intermediate_size,
                "activation": "silu",
                "coupled_roles": "w1_w3_post_swiglu_w2",
            }
            if heterogeneous_layout is not None:
                mlp_attributes.update(
                    {
                        "external_hidden_size": contract.hidden_size,
                        "selection_evidence_sha256": (
                            contract.selection_evidence_sha256
                        ),
                        "sealed_heterogeneous_physical_geometry": True,
                    }
                )
            mlp = builder.add_node(
                f"{prefix}.mlp",
                CircuitKind.GATED_MLP,
                mlp_path,
                layer_index=index,
                inputs=(
                    _port(
                        "residual",
                        PortKind.NORMALIZED_RESIDUAL,
                        "batch",
                        "sequence",
                        hidden,
                    ),
                ),
                outputs=(
                    _port(
                        "delta",
                        PortKind.BRANCH_DELTA,
                        "batch",
                        "sequence",
                        hidden,
                    ),
                ),
                role_prefix="mlp",
                attributes=mlp_attributes,
            )
            mlp_add = builder.add_node(
                f"{prefix}.mlp_residual",
                CircuitKind.RESIDUAL_ADD,
                layer_path,
                layer_index=index,
                inputs=(
                    _port("base", PortKind.RESIDUAL, "batch", "sequence", hidden),
                    _port("delta", PortKind.BRANCH_DELTA, "batch", "sequence", hidden),
                ),
                outputs=(
                    _port("residual", PortKind.RESIDUAL, "batch", "sequence", hidden),
                ),
                attributes={"ordered_branch": "mlp"},
            )
            builder.edge(operator_add, "residual", ffn_norm, "residual", EdgeKind.DATA)
            builder.edge(ffn_norm, "normalized", mlp, "residual", EdgeKind.DATA)
            builder.edge(operator_add, "residual", mlp_add, "base", EdgeKind.RESIDUAL)
            builder.edge(mlp, "delta", mlp_add, "delta", EdgeKind.RESIDUAL)
            current = mlp_add

        final_norm = builder.add_node(
            "output.final_norm",
            CircuitKind.NORM,
            final_norm_path,
            layer_index=None,
            inputs=(_port("residual", PortKind.RESIDUAL, "batch", "sequence", hidden),),
            outputs=(
                _port(
                    "normalized",
                    PortKind.NORMALIZED_RESIDUAL,
                    "batch",
                    "sequence",
                    hidden,
                ),
            ),
            role_prefix="final_norm",
            attributes={"norm_kind": "rms_norm"},
        )
        head = builder.add_node(
            "output.lm_head",
            CircuitKind.OUTPUT_HEAD,
            head_path,
            layer_index=None,
            inputs=(
                _port(
                    "residual",
                    PortKind.NORMALIZED_RESIDUAL,
                    "batch",
                    "sequence",
                    hidden,
                ),
            ),
            outputs=(
                _port(
                    "logits",
                    PortKind.LOGITS,
                    "batch",
                    "sequence",
                    vocab,
                    modality="text",
                ),
            ),
            role_prefix="output_head",
            attributes={
                "vocabulary_size": vocab,
                "tied_to_embedding": bool(
                    getattr(model.config, "tie_word_embeddings", False)
                ),
            },
        )
        builder.edge(current, "residual", final_norm, "residual", EdgeKind.DATA)
        builder.edge(final_norm, "normalized", head, "residual", EdgeKind.DATA)
        builder.interfaces.append(
            ModalityInterface(
                interface_id="text.logits.output",
                modality="text",
                direction="output",
                node_id=head.node_id,
                port_name="logits",
                representation="token_logits",
            )
        )
        hybrid_markers = _validated_lfm2_hybrid_markers(
            model.config,
            layer_count=len(layers),
        )
        metadata: dict[str, Any] = {
            "inspector": self.name,
            "position_policy": "rope",
            "layer_count": len(layers),
            "hidden_size": hidden,
            "hybrid_attention_short_conv": True,
            **hybrid_markers,
        }
        if heterogeneous_layout is not None:
            assert heterogeneous_receipt is not None
            audit = heterogeneous_receipt["physical_materialization_audit"]
            metadata.update(
                {
                    "physical_layout": "heterogeneous_dense_fp32",
                    "physical_layout_descriptor_sha256": (
                        heterogeneous_layout.descriptor_sha256
                    ),
                    "physical_layout_evidence_binding_sha256": (
                        heterogeneous_layout.evidence_binding_sha256
                    ),
                    "physical_materialization_audit_sha256": audit[
                        "result_sha256"
                    ],
                    "blank_materialization_receipt_sha256": (
                        heterogeneous_receipt["receipt_sha256"]
                    ),
                    "physical_parameter_state_key_count": (
                        heterogeneous_receipt["parameter_state_key_count"]
                    ),
                    "runtime_parameter_alias_group_count": 0,
                    "cross_layer_parameter_sharing": False,
                    "runtime_bridge_parameter_elements": 0,
                    "runtime_sidecar_parameter_elements": 0,
                    "parameters_quantized": 0,
                    "sparse_parameter_storage": False,
                    "pruning_masks_present": False,
                }
            )
        graph = builder.finish(metadata=metadata)
        if heterogeneous_layout is not None and (
            graph.logical_parameter_count
            != heterogeneous_layout.accounting.logical_parameter_elements
            or graph.unique_parameter_count
            != heterogeneous_layout.accounting.unique_parameter_elements
            or graph.logical_parameter_count != graph.unique_parameter_count
            or graph.unique_parameter_bytes
            != heterogeneous_layout.accounting.parameter_bytes_fp32
        ):
            raise DynamicModelInspectionError(
                "Heterogeneous LFM2 graph accounting disagrees with its sealed layout."
            )
        return graph


class InspectorRegistry:
    """Deterministic priority-ordered model inspector registry."""

    def __init__(
        self,
        inspectors: Sequence[ModelCircuitInspector] | None = None,
    ) -> None:
        self._inspectors: dict[str, ModelCircuitInspector] = {}
        for inspector in inspectors or ():
            self.register(inspector)

    def register(self, inspector: ModelCircuitInspector) -> None:
        name = getattr(inspector, "name", None)
        priority = getattr(inspector, "priority", None)
        if not isinstance(name, str) or not name.strip():
            raise DynamicModelInspectionError("Inspector name must be nonempty.")
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise DynamicModelInspectionError("Inspector priority must be an integer.")
        if name in self._inspectors:
            raise DynamicModelInspectionError(
                f"Inspector {name!r} is already registered."
            )
        self._inspectors[name] = inspector

    @property
    def inspectors(self) -> tuple[ModelCircuitInspector, ...]:
        return tuple(
            sorted(
                self._inspectors.values(),
                key=lambda value: (-value.priority, value.name),
            )
        )

    def select(self, model: Any) -> ModelCircuitInspector:
        supported = [value for value in self.inspectors if value.supports(model)]
        if not supported:
            model_type = getattr(getattr(model, "config", None), "model_type", None)
            raise DynamicModelInspectionError(
                "No registered inspector supports model_type "
                f"{model_type!r}; install or implement a topology adapter."
            )
        highest = supported[0].priority
        winners = [value for value in supported if value.priority == highest]
        if len(winners) != 1:
            raise DynamicModelInspectionError(
                "Multiple inspectors have equal highest priority: "
                + ", ".join(value.name for value in winners)
            )
        return winners[0]

    def inspect(
        self,
        model: Any,
        *,
        model_name: str,
        tokenizer_fingerprint: str | None = None,
    ) -> ModelCircuitGraph:
        return self.select(model).inspect(
            model,
            model_name=model_name,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )


def default_inspector_registry() -> InspectorRegistry:
    return InspectorRegistry(
        (
            Lfm2CircuitInspector(),
            Qwen3CodebookCircuitInspector(),
            Qwen3CompensatedCircuitInspector(),
            NeoNativeVisionCircuitInspector(),
            NativeAudioCircuitInspector(),
            HFCausalDecoderInspector(),
        )
    )


def inspect_model_circuits(
    model: Any,
    *,
    model_name: str,
    tokenizer_fingerprint: str | None = None,
    registry: InspectorRegistry | None = None,
) -> ModelCircuitGraph:
    """Compile one concrete model into the canonical causal-circuit IR."""

    selected = registry or default_inspector_registry()
    return selected.inspect(
        model,
        model_name=model_name,
        tokenizer_fingerprint=tokenizer_fingerprint,
    )


__all__ = [
    "NATIVE_AUDIO_FAMILY",
    "NATIVE_AUDIO_LANE_CONFIG_KEY",
    "NATIVE_AUDIO_LANE_MARKER",
    "NATIVE_AUDIO_MODEL_TYPE",
    "NATIVE_AUDIO_REPRESENTATION",
    "DynamicModelInspectionError",
    "HFCausalDecoderInspector",
    "InspectorRegistry",
    "Lfm2CircuitInspector",
    "ModelCircuitInspector",
    "NativeAudioCircuitInspector",
    "NeoNativeVisionCircuitInspector",
    "Qwen3CodebookCircuitInspector",
    "Qwen3CompensatedCircuitInspector",
    "default_inspector_registry",
    "inspect_model_circuits",
]
