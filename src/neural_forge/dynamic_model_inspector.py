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


def _disable_torch_function_subclass(torch: Any) -> Any:
    guard = getattr(
        getattr(torch, "_C", None),
        "DisableTorchFunctionSubclass",
        None,
    )
    if guard is None:
        raise DynamicModelInspectionError(
            "PyTorch cannot disable Tensor-subclass dispatch for authenticated "
            "physical-state inspection."
        )
    return guard()


def _storage_groups(model: Any) -> dict[str, str]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - model inspection requires torch
        raise DynamicModelInspectionError(
            "Dynamic inspection requires PyTorch."
        ) from exc

    groups: dict[tuple[Any, ...], str] = {}
    result: dict[str, str] = {}
    entries: list[
        tuple[
            str,
            tuple[Any, ...],
            tuple[Any, ...],
            tuple[int, int],
        ]
    ] = []
    for name, parameter in sorted(
        model.named_parameters(recurse=True, remove_duplicate=False)
    ):
        try:
            if not isinstance(parameter, torch.Tensor):
                raise DynamicModelInspectionError(
                    f"Model parameter {name!r} is not a PyTorch tensor."
                )
            # Use unbound base-Tensor accessors.  Tensor subclasses can override
            # instance methods such as ``untyped_storage`` and ``data_ptr``;
            # trusting those overrides would let two unrelated allocations claim
            # one physical identity (or hide a real cross-StorageImpl alias).
            with _disable_torch_function_subclass(torch):
                device = torch.Tensor.device.__get__(parameter, torch.Tensor)
                dtype = torch.Tensor.dtype.__get__(parameter, torch.Tensor)
                storage = torch.Tensor.untyped_storage(parameter)
                storage_ptr = int(storage.data_ptr())
                tensor_ptr = int(torch.Tensor.data_ptr(parameter))
                storage_nbytes = int(storage.nbytes())
                strides = tuple(int(value) for value in torch.Tensor.stride(parameter))
                shape = tuple(int(value) for value in torch.Tensor.size(parameter))
                offset = int(torch.Tensor.storage_offset(parameter))
                element_size = int(torch.Tensor.element_size(parameter))
            if device.type == "meta":
                raise DynamicModelInspectionError(
                    "Dynamic inspection requires materialized parameters."
                )
            storage_token = getattr(storage, "_cdata", None)
            if (
                isinstance(storage_token, bool)
                or not isinstance(storage_token, int)
                or storage_token <= 0
            ):
                raise DynamicModelInspectionError(
                    f"Could not establish a valid parameter storage identity for "
                    f"{name!r}."
                )
            if any(dimension <= 0 for dimension in shape):
                raise DynamicModelInspectionError(
                    f"Zero-sized parameter {name!r} is not representable by the "
                    "circuit IR."
                )
            covered_span = 1
            for stride, dimension in sorted(
                (abs(stride), dimension)
                for stride, dimension in zip(strides, shape, strict=True)
                if dimension > 1
            ):
                if stride < covered_span:
                    raise DynamicModelInspectionError(
                        f"Internally overlapping parameter {name!r} is not "
                        "representable by the circuit IR."
                    )
                covered_span += (dimension - 1) * stride
            minimum = offset + sum(
                min(0, (dimension - 1) * stride)
                for dimension, stride in zip(shape, strides, strict=True)
            )
            maximum = offset + sum(
                max(0, (dimension - 1) * stride)
                for dimension, stride in zip(shape, strides, strict=True)
            )
            relative_span = (
                minimum * element_size,
                (maximum + 1) * element_size,
            )
            if (
                storage_ptr <= 0
                or tensor_ptr <= 0
                or storage_nbytes <= 0
                or element_size <= 0
                or tensor_ptr != storage_ptr + offset * element_size
                or relative_span[0] < 0
                or relative_span[1] > storage_nbytes
                or relative_span[0] >= relative_span[1]
            ):
                raise DynamicModelInspectionError(
                    f"Could not authenticate the physical parameter span for {name!r}."
                )
            device_identity = (device.type, device.index)
            absolute_span = (
                storage_ptr + relative_span[0],
                storage_ptr + relative_span[1],
            )
            # Exact aliases are physical views, not StorageImpl identities.
            # Separate torch.from_numpy calls can wrap the same bytes with
            # different StorageImpl objects; conversely a Tensor subclass can
            # spoof a positive ``_cdata`` value.  Absolute authenticated spans
            # make both cases unambiguous.
            key = (
                *device_identity,
                str(dtype),
                tensor_ptr,
                shape,
                strides,
            )
        except (AttributeError, RuntimeError) as exc:
            raise DynamicModelInspectionError(
                f"Could not establish parameter storage identity for {name!r}."
            ) from exc
        entries.append((name, device_identity, key, absolute_span))

    # The IR can represent exact aliases and independent storages, but not two
    # learned parameters that cover only part of the same physical allocation.
    # Treating such views as independent would overcount the child and allow a
    # component transaction to mutate an unowned alias.  Bounding-span overlap
    # is intentionally conservative for strided views: an unfamiliar layout
    # fails closed instead of receiving a false exact-compaction certificate.
    entries_by_device: dict[
        tuple[Any, ...],
        list[tuple[str, tuple[Any, ...], tuple[Any, ...], tuple[int, int]]],
    ] = {}
    for entry in entries:
        entries_by_device.setdefault(entry[1], []).append(entry)
    for device_entries in entries_by_device.values():
        # Collapse exact aliases before sweeping their conservative byte spans.
        # Sorting by interval endpoints and path makes both the result and any
        # fail-closed diagnostic independent of registration order.  The prior
        # interval with the furthest end is sufficient: if it does not overlap
        # the current interval, no earlier interval can overlap it either.
        representatives: dict[
            tuple[Any, ...],
            tuple[str, tuple[Any, ...], tuple[Any, ...], tuple[int, int]],
        ] = {}
        for entry in device_entries:
            representatives.setdefault(entry[2], entry)
        ordered = sorted(
            representatives.values(),
            key=lambda entry: (entry[3][0], entry[3][1], entry[0]),
        )
        furthest_name: str | None = None
        furthest_end: int | None = None
        for name, _device, _key, span in ordered:
            if furthest_end is not None and span[0] < furthest_end:
                raise DynamicModelInspectionError(
                    "Partially overlapping parameter storage is not representable "
                    f"by the circuit IR: {furthest_name!r} and {name!r}."
                )
            if furthest_end is None or span[1] > furthest_end:
                furthest_name = name
                furthest_end = span[1]

    for name, _device, key, _span in entries:
        group = groups.setdefault(key, f"storage-{len(groups):06d}")
        result[name] = group
    return result


def _parameter_counts(model: Any) -> tuple[int, int, int]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - model inspection requires torch
        raise DynamicModelInspectionError(
            "Dynamic inspection requires PyTorch."
        ) from exc

    logical = 0
    unique: dict[str, tuple[int, int]] = {}
    groups = _storage_groups(model)
    for name, value in model.named_parameters(recurse=True, remove_duplicate=False):
        if not isinstance(value, torch.Tensor):
            raise DynamicModelInspectionError(
                f"Model parameter {name!r} is not a PyTorch tensor."
            )
        with _disable_torch_function_subclass(torch):
            count = int(torch.Tensor.numel(value))
            size = int(torch.Tensor.element_size(value))
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


NATIVE_VIDEO_LANE_CONFIG_KEY = "neural_forge_native_video_lane"
NATIVE_VIDEO_LANE_MARKER = "neural-forge-native-video-lane-v1"
NATIVE_VIDEO_MODEL_TYPE = "native_video_chat"
NATIVE_VIDEO_FAMILY = "native_video_chat"
NATIVE_VIDEO_REPRESENTATION = "rgb24_frames"


class NativeVideoCircuitInspector:
    """Compile a learned RGB24 tubelet encoder plus its native text decoder.

    This is the video mirror of :class:`NativeAudioCircuitInspector`: the clip
    enters through one ``MODALITY_INPUT`` node whose input port is a raw
    ``rgb24_frames`` modality port, the encoder result is projected into the
    decoder residual stream, and the text decoder is compiled exactly as the
    audio and vision lanes compile theirs.  There is no frozen external
    encoder anywhere in the graph: every video tensor is a learned parameter
    this inspector owns.

    One strided ``Conv3d`` performs the per-frame patch embedding and the
    temporal aggregation together -- a tubelet -- which is the direct analog of
    the audio lane's strided ``Conv1d`` frame encoder and NEO's ``Conv2d``
    patch embedding.  The temporal stride is what makes this a video front end
    rather than a per-frame image front end, so it is validated rather than
    assumed.

    ``supports`` is deliberately keyed on an explicit opt-in marker rather
    than on a model type alone.  A model must set
    ``config.neural_forge_native_video_lane = NATIVE_VIDEO_LANE_MARKER`` to be
    routed here, so the inspector can never hijack a real video-capable model
    that has not declared this exact topology.
    """

    name = "native-video"
    priority = 240

    def supports(self, model: Any) -> bool:
        config = getattr(model, "config", None)
        return bool(
            getattr(config, NATIVE_VIDEO_LANE_CONFIG_KEY, None)
            == NATIVE_VIDEO_LANE_MARKER
            and getattr(config, "model_type", None) == NATIVE_VIDEO_MODEL_TYPE
            and hasattr(model, "video_encoder")
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
                "Native video inspection requires PyTorch."
            ) from exc

        video_modality = canonical_modality("video", "native video input modality")
        text_modality = canonical_modality("text", "native video text modality")
        config = getattr(model, "config", None)
        video_config = getattr(config, "video_config", None)
        decoder_config = getattr(config, "decoder_config", None)
        video_encoder = getattr(model, "video_encoder", None)
        language_model = getattr(model, "language_model", None)
        if (
            config is None
            or getattr(config, NATIVE_VIDEO_LANE_CONFIG_KEY, None)
            != NATIVE_VIDEO_LANE_MARKER
            or getattr(config, "model_type", None) != NATIVE_VIDEO_MODEL_TYPE
            or video_config is None
            or decoder_config is None
            or video_encoder is None
            or language_model is None
        ):
            raise DynamicModelInspectionError(
                "Native video inspection requires one marked native_video_chat "
                "config with video_config, decoder_config, video_encoder, and "
                "language_model."
            )

        tubelet_encoder = getattr(video_encoder, "tubelet_conv", None)
        frame_projection = getattr(video_encoder, "projection", None)
        if not isinstance(tubelet_encoder, nn.Conv3d) or not isinstance(
            frame_projection, nn.Linear
        ):
            raise DynamicModelInspectionError(
                "Native video front end must expose a Conv3d tubelet_conv and a "
                "Linear projection."
            )

        try:
            frame_count = int(
                _singleton_config_scalar(
                    video_config.num_frames,
                    "native video num_frames",
                )
            )
            frame_height = int(
                _singleton_config_scalar(
                    video_config.frame_height,
                    "native video frame_height",
                )
            )
            frame_width = int(
                _singleton_config_scalar(
                    video_config.frame_width,
                    "native video frame_width",
                )
            )
            patch_size = int(
                _singleton_config_scalar(
                    video_config.patch_size,
                    "native video patch_size",
                )
            )
            temporal_patch_size = int(
                _singleton_config_scalar(
                    video_config.temporal_patch_size,
                    "native video temporal_patch_size",
                )
            )
            input_channels = int(
                _singleton_config_scalar(
                    video_config.num_channels,
                    "native video num_channels",
                )
            )
            fps = int(_singleton_config_scalar(video_config.fps, "native video fps"))
            video_hidden = int(video_config.hidden_size)
            decoder_hidden = int(decoder_config.hidden_size)
        except DynamicModelInspectionError:
            raise
        except (AttributeError, TypeError, ValueError) as exc:
            raise DynamicModelInspectionError(
                "Native video front-end/decoder dimensions are incomplete."
            ) from exc
        if any(
            value <= 0
            for value in (
                frame_count,
                frame_height,
                frame_width,
                patch_size,
                temporal_patch_size,
                input_channels,
                fps,
                video_hidden,
                decoder_hidden,
            )
        ):
            raise DynamicModelInspectionError(
                "Native video front-end/decoder dimensions must be positive."
            )
        frame_encoding = getattr(video_config, "frame_encoding", None)
        if frame_encoding != "uint8_normalized_rgb":
            raise DynamicModelInspectionError(
                "Native video front end must consume uint8-normalized RGB "
                "frames rather than a precomputed external feature."
            )
        if (
            frame_count % temporal_patch_size
            or frame_height % patch_size
            or frame_width % patch_size
        ):
            raise DynamicModelInspectionError(
                "Native video clip geometry must divide exactly into tubelets."
            )
        if temporal_patch_size > frame_count:
            raise DynamicModelInspectionError(
                "Native video temporal patch exceeds the declared clip length."
            )
        tubelet_shape = (temporal_patch_size, patch_size, patch_size)
        if (
            tubelet_encoder.in_channels != input_channels
            or tubelet_encoder.out_channels != video_hidden
            or tubelet_encoder.kernel_size != tubelet_shape
            or tubelet_encoder.stride != tubelet_shape
        ):
            raise DynamicModelInspectionError(
                "Native video tubelet encoder differs from the declared strided "
                "spatio-temporal patch topology."
            )
        video_tokens = (
            (frame_count // temporal_patch_size)
            * (frame_height // patch_size)
            * (frame_width // patch_size)
        )
        if (
            tuple(frame_projection.weight.shape) != (decoder_hidden, video_hidden)
            or frame_projection.bias is not None
        ):
            raise DynamicModelInspectionError(
                "Native video projection must be a bias-free Linear from the "
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
                "Native video decoder topology dimensions must be positive."
            )
        if (
            query_heads % key_value_heads
            or query_heads * head_dim != decoder_hidden
            or getattr(decoder_config, "hidden_act", None) != "silu"
            or getattr(decoder_config, "attention_bias", None) is not False
        ):
            raise DynamicModelInspectionError(
                "Native video decoder must be a bias-free SwiGLU grouped-query "
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
                "Native video decoder is missing its canonical token embedding, "
                "learned position embedding, layers, final norm, or output head."
            )
        if len(decoder_layers) != expected_layers:
            raise DynamicModelInspectionError(
                "Native video decoder depth differs from decoder_config."
            )
        if video_tokens + 1 > max_positions:
            raise DynamicModelInspectionError(
                "Native video token count leaves no learned position for text."
            )
        if (
            tuple(token_embedding.weight.shape) != (vocab_size, decoder_hidden)
            or tuple(position_embedding.weight.shape)
            != (max_positions, decoder_hidden)
            or tuple(lm_head_module.weight.shape) != (vocab_size, decoder_hidden)
            or lm_head_module.bias is not None
        ):
            raise DynamicModelInspectionError(
                "Native video token/position embedding or output head shapes "
                "are incompatible."
            )
        if bool(getattr(decoder_config, "tie_word_embeddings", False)) and (
            token_embedding.weight is not lm_head_module.weight
        ):
            raise DynamicModelInspectionError(
                "Native video config requests tied token/output weights but "
                "runtime storage is not tied."
            )

        groups = _storage_groups(model)
        builder = _GraphBuilder(
            model=model,
            model_name=model_name,
            family=NATIVE_VIDEO_FAMILY,
            tokenizer_fingerprint=tokenizer_fingerprint,
            groups=groups,
            nodes=[],
            edges=[],
            interfaces=[],
            captured=set(),
        )
        video_path = _module_path(model, video_encoder)
        tubelet_path = _module_path(model, tubelet_encoder)
        projection_path = _module_path(model, frame_projection)
        language_path = _module_path(model, language_model)

        tubelets = builder.add_node(
            "input.video.rgb24_tubelet_encoder",
            CircuitKind.MODALITY_INPUT,
            tubelet_path,
            layer_index=None,
            inputs=(
                _port(
                    "clip",
                    PortKind.MODALITY,
                    "batch",
                    "channels",
                    "frames",
                    "height",
                    "width",
                    modality=video_modality,
                ),
            ),
            outputs=(
                _port(
                    "tubelets",
                    PortKind.MODALITY,
                    "batch",
                    "video_tokens",
                    video_hidden,
                    modality=video_modality,
                ),
            ),
            role_prefix="video_tubelet_encoder",
            attributes={
                "input_channels": input_channels,
                "frame_count": frame_count,
                "frame_height": frame_height,
                "frame_width": frame_width,
                "patch_size": patch_size,
                "temporal_patch_size": temporal_patch_size,
                "video_token_count": video_tokens,
                "fps": fps,
                "frame_encoding": "uint8_normalized_rgb",
                "video_hidden_size": video_hidden,
                "activation": "gelu",
                "external_frozen_encoder": False,
            },
        )
        video_projection = builder.add_node(
            "input.video.dense_projection",
            CircuitKind.INTERFACE_PROJECTION,
            projection_path,
            layer_index=None,
            inputs=(
                _port(
                    "tubelets",
                    PortKind.MODALITY,
                    "batch",
                    "video_tokens",
                    video_hidden,
                    modality=video_modality,
                ),
            ),
            outputs=(
                _port(
                    "video_embeddings",
                    PortKind.MODALITY,
                    "video_tokens",
                    decoder_hidden,
                    modality=video_modality,
                ),
            ),
            role_prefix="video_dense_embedding",
            attributes={
                "decoder_hidden_size": decoder_hidden,
                "video_hidden_size": video_hidden,
            },
        )
        builder.edge(
            tubelets,
            "tubelets",
            video_projection,
            "tubelets",
            EdgeKind.MODALITY,
        )
        builder.interfaces.append(
            ModalityInterface(
                interface_id="video.rgb24_frames.input",
                modality=video_modality,
                direction="input",
                node_id=tubelets.node_id,
                port_name="clip",
                representation=NATIVE_VIDEO_REPRESENTATION,
            )
        )

        captured_video = {
            path for path in builder.captured if path.startswith(f"{video_path}.")
        }
        expected_video = {
            path
            for path, _parameter in model.named_parameters(
                recurse=True,
                remove_duplicate=False,
            )
            if path.startswith(f"{video_path}.")
        }
        if captured_video != expected_video:
            raise DynamicModelInspectionError(
                "Native video front end contains learned parameters outside "
                "its tubelet encoder and dense projection operators."
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
                    f"Native video decoder {label} must be a bias-free "
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
                    f"Native video decoder {label} must own one width-{width} "
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
            "input.video_text_embedding_fusion",
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
                    "video_embeddings",
                    PortKind.MODALITY,
                    "video_tokens",
                    decoder_hidden,
                    modality=video_modality,
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
                "operation": "video_context_tokens_in_decoder_residual_stream",
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
            video_projection,
            "video_embeddings",
            fusion,
            "video_embeddings",
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
                    f"Native video decoder layer {layer_index} is missing "
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
                    f"Native video decoder layer {layer_index} attention owns "
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
                    f"Native video decoder layer {layer_index} MLP owns "
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

        video_parameter_count = sum(
            int(parameter.numel())
            for path, parameter in model.named_parameters(
                recurse=True,
                remove_duplicate=False,
            )
            if path.startswith(f"{video_path}.")
        )
        return builder.finish(
            metadata={
                "inspector": self.name,
                "decoder_family": "native_video_text_decoder",
                "layer_count": expected_layers,
                "hidden_size": decoder_hidden,
                "decoder_intermediate_size": intermediate_size,
                "video_hidden_size": video_hidden,
                "video_parameter_count": video_parameter_count,
                "video_frame_count": frame_count,
                "video_frame_height": frame_height,
                "video_frame_width": frame_width,
                "video_patch_size": patch_size,
                "video_temporal_patch_size": temporal_patch_size,
                "video_token_count": video_tokens,
                "video_fps": fps,
                "native_rgb24_tubelet_encoder": True,
                "video_token_replacement_fusion": True,
                "external_frozen_video_encoder": False,
            }
        )


NATIVE_POINTS3D_LANE_CONFIG_KEY = "neural_forge_native_points3d_lane"
NATIVE_POINTS3D_LANE_MARKER = "neural-forge-native-points3d-lane-v1"
NATIVE_POINTS3D_MODEL_TYPE = "native_points3d_chat"
NATIVE_POINTS3D_FAMILY = "native_points3d_chat"
NATIVE_POINTS3D_REPRESENTATION = "quantized_xyz"
NATIVE_POINTS3D_EMITTED_REPRESENTATION = "quantized_xyz_centroid"
NATIVE_POINTS3D_POOLINGS = ("max_over_points", "mean_over_points", "sum_over_points")


class NativePoints3dCircuitInspector:
    """Compile a learned quantized point-set encoder plus its text decoder.

    This is the points3d mirror of :class:`NativeVideoCircuitInspector`: the
    cloud enters through one ``MODALITY_INPUT`` node whose input port is a raw
    ``quantized_xyz`` modality port, the encoder result is projected into the
    decoder residual stream, and the text decoder is compiled exactly as the
    video, audio, and vision lanes compile theirs.  There is no frozen external
    encoder anywhere in the graph: every point tensor is a learned parameter
    this inspector owns.

    A point set has no grid, so the front end cannot be a convolution.  It is
    the standard point-set shape instead: one shared ``Linear`` applied to every
    point independently, then one **symmetric** reduction over the point axis
    into ``num_tokens`` context tokens.  Permutation invariance is the property
    that makes this a point-*set* encoder rather than a sequence encoder over an
    accidental storage order, so the declared reduction is validated against
    :data:`NATIVE_POINTS3D_POOLINGS` rather than assumed -- the direct analog of
    the video lane validating that its ``Conv3d`` stride is genuinely temporal.

    ``supports`` is deliberately keyed on an explicit opt-in marker rather than
    on a model type alone.  A model must set
    ``config.neural_forge_native_points3d_lane = NATIVE_POINTS3D_LANE_MARKER``
    to be routed here, so the inspector can never hijack a real geometry-capable
    model that has not declared this exact topology.

    One optional arm is the repository's first ``MODALITY_OUTPUT`` emission.
    When ``points_config.emit_centroid`` is true the model must own a bias-free
    ``centroid_head`` whose output is a quantized XYZ triple, and the graph
    gains a ``CircuitKind.MODALITY_OUTPUT`` node plus a ``direction="output"``
    :class:`ModalityInterface` whose modality is **not** text.  The arm is
    additive in both directions: a model without the head compiles to exactly
    the input-side graph this lane shipped with.
    """

    name = "native-points3d"
    priority = 240

    def supports(self, model: Any) -> bool:
        config = getattr(model, "config", None)
        return bool(
            getattr(config, NATIVE_POINTS3D_LANE_CONFIG_KEY, None)
            == NATIVE_POINTS3D_LANE_MARKER
            and getattr(config, "model_type", None) == NATIVE_POINTS3D_MODEL_TYPE
            and hasattr(model, "point_encoder")
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
                "Native points3d inspection requires PyTorch."
            ) from exc

        points_modality = canonical_modality(
            "points3d",
            "native points3d input modality",
        )
        text_modality = canonical_modality("text", "native points3d text modality")
        config = getattr(model, "config", None)
        points_config = getattr(config, "points_config", None)
        decoder_config = getattr(config, "decoder_config", None)
        point_encoder = getattr(model, "point_encoder", None)
        language_model = getattr(model, "language_model", None)
        if (
            config is None
            or getattr(config, NATIVE_POINTS3D_LANE_CONFIG_KEY, None)
            != NATIVE_POINTS3D_LANE_MARKER
            or getattr(config, "model_type", None) != NATIVE_POINTS3D_MODEL_TYPE
            or points_config is None
            or decoder_config is None
            or point_encoder is None
            or language_model is None
        ):
            raise DynamicModelInspectionError(
                "Native points3d inspection requires one marked "
                "native_points3d_chat config with points_config, decoder_config, "
                "point_encoder, and language_model."
            )

        point_linear = getattr(point_encoder, "point_linear", None)
        point_projection = getattr(point_encoder, "projection", None)
        if not isinstance(point_linear, nn.Linear) or not isinstance(
            point_projection,
            nn.Linear,
        ):
            raise DynamicModelInspectionError(
                "Native points3d front end must expose a Linear point_linear and "
                "a Linear projection."
            )

        try:
            point_count = int(
                _singleton_config_scalar(
                    points_config.num_points,
                    "native points3d num_points",
                )
            )
            coordinates = int(
                _singleton_config_scalar(
                    points_config.num_coordinates,
                    "native points3d num_coordinates",
                )
            )
            point_tokens = int(
                _singleton_config_scalar(
                    points_config.num_tokens,
                    "native points3d num_tokens",
                )
            )
            points_hidden = int(points_config.hidden_size)
            decoder_hidden = int(decoder_config.hidden_size)
        except DynamicModelInspectionError:
            raise
        except (AttributeError, TypeError, ValueError) as exc:
            raise DynamicModelInspectionError(
                "Native points3d front-end/decoder dimensions are incomplete."
            ) from exc
        if any(
            value <= 0
            for value in (
                point_count,
                coordinates,
                point_tokens,
                points_hidden,
                decoder_hidden,
            )
        ):
            raise DynamicModelInspectionError(
                "Native points3d front-end/decoder dimensions must be positive."
            )
        if coordinates != 3:
            raise DynamicModelInspectionError(
                "Native points3d front end must consume three-dimensional XYZ "
                "coordinates."
            )
        point_encoding = getattr(points_config, "point_encoding", None)
        if point_encoding != "int16_quantized_xyz":
            raise DynamicModelInspectionError(
                "Native points3d front end must consume dequantized int16 XYZ "
                "coordinates rather than a precomputed external feature."
            )
        pooling = getattr(points_config, "pooling", None)
        if pooling not in NATIVE_POINTS3D_POOLINGS:
            raise DynamicModelInspectionError(
                "Native points3d front end must declare a symmetric pooling in: "
                + ", ".join(NATIVE_POINTS3D_POOLINGS)
                + "; an order-dependent reduction is not a point-set encoder."
            )
        if (
            point_linear.in_features != coordinates
            or point_linear.out_features != point_tokens * points_hidden
        ):
            raise DynamicModelInspectionError(
                "Native points3d per-point Linear must map three coordinates "
                "onto exactly num_tokens * hidden_size pooled features."
            )
        if (
            tuple(point_projection.weight.shape) != (decoder_hidden, points_hidden)
            or point_projection.bias is not None
        ):
            raise DynamicModelInspectionError(
                "Native points3d projection must be a bias-free Linear from the "
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
                "Native points3d decoder topology dimensions must be positive."
            )
        if (
            query_heads % key_value_heads
            or query_heads * head_dim != decoder_hidden
            or getattr(decoder_config, "hidden_act", None) != "silu"
            or getattr(decoder_config, "attention_bias", None) is not False
        ):
            raise DynamicModelInspectionError(
                "Native points3d decoder must be a bias-free SwiGLU grouped-query "
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
                "Native points3d decoder is missing its canonical token "
                "embedding, learned position embedding, layers, final norm, or "
                "output head."
            )
        if len(decoder_layers) != expected_layers:
            raise DynamicModelInspectionError(
                "Native points3d decoder depth differs from decoder_config."
            )
        if point_tokens + 1 > max_positions:
            raise DynamicModelInspectionError(
                "Native points3d token count leaves no learned position for text."
            )
        if (
            tuple(token_embedding.weight.shape) != (vocab_size, decoder_hidden)
            or tuple(position_embedding.weight.shape)
            != (max_positions, decoder_hidden)
            or tuple(lm_head_module.weight.shape) != (vocab_size, decoder_hidden)
            or lm_head_module.bias is not None
        ):
            raise DynamicModelInspectionError(
                "Native points3d token/position embedding or output head shapes "
                "are incompatible."
            )
        if bool(getattr(decoder_config, "tie_word_embeddings", False)) and (
            token_embedding.weight is not lm_head_module.weight
        ):
            raise DynamicModelInspectionError(
                "Native points3d config requests tied token/output weights but "
                "runtime storage is not tied."
            )

        emit_centroid = bool(getattr(points_config, "emit_centroid", False))
        centroid_head = getattr(model, "centroid_head", None)
        if emit_centroid:
            if (
                not isinstance(centroid_head, nn.Linear)
                or tuple(centroid_head.weight.shape) != (coordinates, decoder_hidden)
                or centroid_head.bias is not None
            ):
                raise DynamicModelInspectionError(
                    "Native points3d emitted-centroid head must be a bias-free "
                    "Linear from the decoder hidden size onto three coordinates."
                )
        elif centroid_head is not None:
            raise DynamicModelInspectionError(
                "Native points3d model owns a centroid_head without declaring "
                "points_config.emit_centroid; an unreviewed emission path is not "
                "an inspectable graph."
            )

        groups = _storage_groups(model)
        builder = _GraphBuilder(
            model=model,
            model_name=model_name,
            family=NATIVE_POINTS3D_FAMILY,
            tokenizer_fingerprint=tokenizer_fingerprint,
            groups=groups,
            nodes=[],
            edges=[],
            interfaces=[],
            captured=set(),
        )
        encoder_path = _module_path(model, point_encoder)
        point_linear_path = _module_path(model, point_linear)
        projection_path = _module_path(model, point_projection)
        language_path = _module_path(model, language_model)

        point_features = builder.add_node(
            "input.points3d.quantized_point_encoder",
            CircuitKind.MODALITY_INPUT,
            point_linear_path,
            layer_index=None,
            inputs=(
                _port(
                    "cloud",
                    PortKind.MODALITY,
                    "batch",
                    "points",
                    coordinates,
                    modality=points_modality,
                ),
            ),
            outputs=(
                _port(
                    "point_tokens",
                    PortKind.MODALITY,
                    "batch",
                    "point_tokens",
                    points_hidden,
                    modality=points_modality,
                ),
            ),
            role_prefix="points3d_point_encoder",
            attributes={
                "coordinates": coordinates,
                "point_count": point_count,
                "point_token_count": point_tokens,
                "point_encoding": "int16_quantized_xyz",
                "points_hidden_size": points_hidden,
                "pooling": pooling,
                "permutation_invariant": True,
                "activation": "gelu",
                "external_frozen_encoder": False,
            },
        )
        point_projection_node = builder.add_node(
            "input.points3d.dense_projection",
            CircuitKind.INTERFACE_PROJECTION,
            projection_path,
            layer_index=None,
            inputs=(
                _port(
                    "point_tokens",
                    PortKind.MODALITY,
                    "batch",
                    "point_tokens",
                    points_hidden,
                    modality=points_modality,
                ),
            ),
            outputs=(
                _port(
                    "point_embeddings",
                    PortKind.MODALITY,
                    "point_tokens",
                    decoder_hidden,
                    modality=points_modality,
                ),
            ),
            role_prefix="points3d_dense_embedding",
            attributes={
                "decoder_hidden_size": decoder_hidden,
                "points_hidden_size": points_hidden,
            },
        )
        builder.edge(
            point_features,
            "point_tokens",
            point_projection_node,
            "point_tokens",
            EdgeKind.MODALITY,
        )
        builder.interfaces.append(
            ModalityInterface(
                interface_id="points3d.quantized_xyz.input",
                modality=points_modality,
                direction="input",
                node_id=point_features.node_id,
                port_name="cloud",
                representation=NATIVE_POINTS3D_REPRESENTATION,
            )
        )

        captured_points = {
            path for path in builder.captured if path.startswith(f"{encoder_path}.")
        }
        expected_points = {
            path
            for path, _parameter in model.named_parameters(
                recurse=True,
                remove_duplicate=False,
            )
            if path.startswith(f"{encoder_path}.")
        }
        if captured_points != expected_points:
            raise DynamicModelInspectionError(
                "Native points3d front end contains learned parameters outside "
                "its per-point encoder and dense projection operators."
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
                    f"Native points3d decoder {label} must be a bias-free "
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
                    f"Native points3d decoder {label} must own one width-{width} "
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
            "input.points3d_text_embedding_fusion",
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
                    "point_embeddings",
                    PortKind.MODALITY,
                    "point_tokens",
                    decoder_hidden,
                    modality=points_modality,
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
                "operation": "point_context_tokens_in_decoder_residual_stream",
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
            point_projection_node,
            "point_embeddings",
            fusion,
            "point_embeddings",
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
                    f"Native points3d decoder layer {layer_index} is missing "
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
                    f"Native points3d decoder layer {layer_index} attention owns "
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
                    f"Native points3d decoder layer {layer_index} MLP owns "
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

        if emit_centroid:
            # The repository's first MODALITY_OUTPUT emission.  It deliberately
            # reads the same normalized residual the text head reads, so the two
            # heads are siblings rather than a chain: an emitted-media head is
            # an additional output of the decoder, not a post-processor of text.
            centroid = builder.add_node(
                "output.points3d.quantized_centroid",
                CircuitKind.MODALITY_OUTPUT,
                _module_path(model, centroid_head),
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
                        "centroid",
                        PortKind.MODALITY,
                        "batch",
                        coordinates,
                        modality=points_modality,
                    ),
                ),
                role_prefix="points3d_centroid_head",
                attributes={
                    "coordinates": coordinates,
                    "emitted_encoding": "int16_quantized_xyz",
                    "reduction": "last_position",
                    "hash_basis_modality": points_modality,
                },
            )
            builder.edge(
                final_norm,
                "normalized",
                centroid,
                "residual",
                EdgeKind.DATA,
            )
            builder.interfaces.append(
                ModalityInterface(
                    interface_id="points3d.quantized_centroid.output",
                    modality=points_modality,
                    direction="output",
                    node_id=centroid.node_id,
                    port_name="centroid",
                    representation=NATIVE_POINTS3D_EMITTED_REPRESENTATION,
                )
            )

        points_parameter_count = sum(
            int(parameter.numel())
            for path, parameter in model.named_parameters(
                recurse=True,
                remove_duplicate=False,
            )
            if path.startswith(f"{encoder_path}.")
        )
        return builder.finish(
            metadata={
                "inspector": self.name,
                "decoder_family": "native_points3d_text_decoder",
                "layer_count": expected_layers,
                "hidden_size": decoder_hidden,
                "decoder_intermediate_size": intermediate_size,
                "points_hidden_size": points_hidden,
                "points_parameter_count": points_parameter_count,
                "points3d_point_count": point_count,
                "points3d_coordinates": coordinates,
                "points3d_token_count": point_tokens,
                "points3d_pooling": pooling,
                "native_quantized_point_set_encoder": True,
                "point_token_replacement_fusion": True,
                "external_frozen_points3d_encoder": False,
                "emits_quantized_centroid": emit_centroid,
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


SMOLVLM_MODEL_TYPE = "smolvlm"
SMOLVLM_FAMILY = "smolvlm"
SMOLVLM_VISION_MODEL_TYPE = "smolvlm_vision"
SMOLVLM_MODEL_CLASS = (
    "transformers.models.smolvlm.modeling_smolvlm.SmolVLMForConditionalGeneration"
)
SMOLVLM_IMAGE_REPRESENTATION = "normalized_rgb_crop_pixels"
SMOLVLM_PATCH_MASK_REPRESENTATION = "patch_attention_mask"
SMOLVLM_VIDEO_NOTE = (
    "video enters as sampled image frames through the same vision tower; this "
    "checkpoint owns no separate video front end, so no video modality "
    "interface is emitted"
)


def _smolvlm_module_paths(model: Any) -> dict[int, str]:
    """Index every child module of ``model`` by identity to its real path."""

    paths: dict[int, str] = {}
    for path, module in model.named_modules():
        if not path:
            continue
        paths.setdefault(id(module), path)
    return paths


class _SmolVlmTextDecoderView:
    """The text decoder presented in the canonical causal-decoder layout.

    ``SmolVLMForConditionalGeneration`` keeps its decoder at
    ``model.text_model`` and its output head at ``lm_head``, which is one
    rename away from the ``model``/``lm_head`` layout every supported causal
    decoder already declares.  This view performs exactly that rename against
    the *same* module objects -- nothing is copied, nothing is re-registered --
    so :class:`HFCausalDecoderInspector` can prove the decoder's topology with
    its own contract instead of a second, drifting transcription of it.
    """

    __slots__ = ("config", "model", "lm_head")

    def __init__(self, model: Any) -> None:
        self.config = model.config.text_config
        self.model = model.model.text_model
        self.lm_head = model.lm_head

    def named_parameters(
        self,
        recurse: bool = True,
        remove_duplicate: bool = False,
    ) -> Any:
        from itertools import chain

        return chain(
            (
                (f"model.{name}", value)
                for name, value in self.model.named_parameters(
                    recurse=recurse, remove_duplicate=remove_duplicate
                )
            ),
            (
                (f"lm_head.{name}", value)
                for name, value in self.lm_head.named_parameters(
                    recurse=recurse, remove_duplicate=remove_duplicate
                )
            ),
        )

    def named_modules(self) -> Any:
        from itertools import chain

        return chain(
            (("model", self.model),),
            (
                (f"model.{name}", module)
                for name, module in self.model.named_modules()
                if name
            ),
            (("lm_head", self.lm_head),),
            (
                (f"lm_head.{name}", module)
                for name, module in self.lm_head.named_modules()
                if name
            ),
        )

    def get_submodule(self, path: str) -> Any:
        head, _, rest = path.partition(".")
        if head not in {"model", "lm_head"}:
            raise AttributeError(path)
        root = self.model if head == "model" else self.lm_head
        return root if not rest else root.get_submodule(rest)


class SmolVlmCircuitInspector:
    """Compile SmolVLM's SigLIP tower, connector and llama text decoder.

    The vision tower and the connector are :class:`CircuitKind.MODALITY_INPUT`
    side operators: every learned tensor between raw normalized RGB crop pixels
    and the decoder-width image token embeddings has exactly one owning node,
    and the inspector refuses any model that hides a learned vision parameter
    somewhere this decomposition does not reach.  The decoder itself is not
    re-derived here -- :class:`HFCausalDecoderInspector` proves it through the
    same layout contract every other llama gets, and this inspector rebases the
    proven subgraph onto the composite model's real module paths.

    Video is *not* a separate lane in this checkpoint.  SmolVLM2 samples video
    into image frames and pushes each frame through this same tower, so the
    honest emission is one image lane, recorded as such in the metadata rather
    than a video interface the weights do not have.
    """

    name = "smolvlm-native-vision"
    priority = 250

    def supports(self, model: Any) -> bool:
        config = getattr(model, "config", None)
        return bool(
            getattr(config, "model_type", None) == SMOLVLM_MODEL_TYPE
            and _qualified_class(model) == SMOLVLM_MODEL_CLASS
        )

    def _decoder_graph(
        self,
        model: Any,
        *,
        model_name: str,
        tokenizer_fingerprint: str | None,
    ) -> ModelCircuitGraph:
        try:
            return HFCausalDecoderInspector().inspect(
                _SmolVlmTextDecoderView(model),
                model_name=f"{model_name}.text_decoder",
                tokenizer_fingerprint=tokenizer_fingerprint,
            )
        except DynamicModelInspectionError as exc:
            raise DynamicModelInspectionError(
                "SmolVLM's text decoder failed the shared causal-decoder "
                f"contract: {exc}"
            ) from exc

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
                "SmolVLM inspection requires PyTorch."
            ) from exc

        if not self.supports(model):
            raise DynamicModelInspectionError(
                "SmolVLM inspection requires one smolvlm "
                "SmolVLMForConditionalGeneration."
            )
        config = model.config
        vision_config = getattr(config, "vision_config", None)
        text_config = getattr(config, "text_config", None)
        composite = getattr(model, "model", None)
        vision_model = getattr(composite, "vision_model", None)
        connector = getattr(composite, "connector", None)
        text_model = getattr(composite, "text_model", None)
        lm_head = getattr(model, "lm_head", None)
        if (
            vision_config is None
            or text_config is None
            or composite is None
            or vision_model is None
            or connector is None
            or text_model is None
            or not isinstance(lm_head, nn.Linear)
        ):
            raise DynamicModelInspectionError(
                "SmolVLM inspection requires vision_config, text_config, "
                "model.vision_model, model.connector, model.text_model and a "
                "Linear lm_head."
            )
        if getattr(vision_config, "model_type", None) != SMOLVLM_VISION_MODEL_TYPE:
            raise DynamicModelInspectionError(
                "SmolVLM vision_config must declare the smolvlm_vision tower."
            )
        if getattr(text_config, "model_type", None) != "llama":
            raise DynamicModelInspectionError(
                "SmolVLM inspection requires a llama text decoder."
            )

        try:
            vision_hidden = int(vision_config.hidden_size)
            vision_intermediate = int(vision_config.intermediate_size)
            vision_layers = int(vision_config.num_hidden_layers)
            vision_heads = int(vision_config.num_attention_heads)
            patch_size = int(vision_config.patch_size)
            image_size = int(vision_config.image_size)
            num_channels = int(vision_config.num_channels)
            layer_norm_eps = float(vision_config.layer_norm_eps)
            vision_activation = str(vision_config.hidden_act)
            scale_factor = int(config.scale_factor)
            image_token_id = int(config.image_token_id)
            decoder_hidden = int(text_config.hidden_size)
        except (AttributeError, TypeError, ValueError) as exc:
            raise DynamicModelInspectionError(
                "SmolVLM vision/connector dimensions are incomplete."
            ) from exc
        if (
            vision_hidden <= 0
            or vision_intermediate <= 0
            or vision_layers <= 0
            or vision_heads <= 0
            or patch_size <= 0
            or image_size <= 0
            or num_channels <= 0
            or scale_factor <= 0
            or decoder_hidden <= 0
            or layer_norm_eps <= 0.0
            or image_size % patch_size
            or vision_hidden % vision_heads
        ):
            raise DynamicModelInspectionError(
                "SmolVLM vision/connector dimensions must be positive and "
                "divide the declared patch grid."
            )
        patches_per_edge = image_size // patch_size
        patch_positions = patches_per_edge * patches_per_edge
        if patches_per_edge % scale_factor:
            raise DynamicModelInspectionError(
                "SmolVLM pixel shuffle requires a patch grid the scale factor divides."
            )
        image_tokens_per_crop = patch_positions // (scale_factor * scale_factor)
        shuffled_width = vision_hidden * scale_factor * scale_factor
        if not 0 <= image_token_id < int(text_config.vocab_size):
            raise DynamicModelInspectionError(
                "SmolVLM image_token_id is outside the decoder vocabulary."
            )

        embeddings = getattr(vision_model, "embeddings", None)
        patch_embedding = getattr(embeddings, "patch_embedding", None)
        position_embedding = getattr(embeddings, "position_embedding", None)
        encoder = getattr(vision_model, "encoder", None)
        encoder_layers = getattr(encoder, "layers", None)
        post_layernorm = getattr(vision_model, "post_layernorm", None)
        modality_projection = getattr(connector, "modality_projection", None)
        projection = getattr(modality_projection, "proj", None)
        if (
            not isinstance(patch_embedding, nn.Conv2d)
            or not isinstance(position_embedding, nn.Embedding)
            or not isinstance(encoder_layers, nn.ModuleList)
            or not isinstance(post_layernorm, nn.LayerNorm)
            or not isinstance(projection, nn.Linear)
        ):
            raise DynamicModelInspectionError(
                "SmolVLM requires a Conv2d patch embedding, a learned position "
                "embedding, a ModuleList encoder, a LayerNorm tower output and "
                "a Linear modality projection."
            )
        if (
            patch_embedding.in_channels != num_channels
            or patch_embedding.out_channels != vision_hidden
            or patch_embedding.kernel_size != (patch_size, patch_size)
            or patch_embedding.stride != (patch_size, patch_size)
            or patch_embedding.bias is None
        ):
            raise DynamicModelInspectionError(
                "SmolVLM patch embedding differs from the declared raw-patch topology."
            )
        if tuple(position_embedding.weight.shape) != (
            patch_positions,
            vision_hidden,
        ):
            raise DynamicModelInspectionError(
                "SmolVLM learned image position embedding does not match the "
                "declared patch grid."
            )
        if len(encoder_layers) != vision_layers:
            raise DynamicModelInspectionError(
                "SmolVLM vision depth differs from vision_config."
            )
        if (
            tuple(projection.weight.shape) != (decoder_hidden, shuffled_width)
            or projection.bias is not None
        ):
            raise DynamicModelInspectionError(
                "SmolVLM modality projection must be a bias-free "
                f"Linear({shuffled_width}, {decoder_hidden})."
            )

        def require_linear(
            module: Any,
            shape: tuple[int, int],
            label: str,
            *,
            bias: bool,
        ) -> None:
            if (
                not isinstance(module, nn.Linear)
                or tuple(module.weight.shape) != shape
                or (module.bias is not None) is not bias
            ):
                state = "with" if bias else "without"
                raise DynamicModelInspectionError(
                    f"SmolVLM vision {label} must be a Linear{shape} {state} bias."
                )

        def require_layer_norm(module: Any, label: str) -> None:
            if (
                not isinstance(module, nn.LayerNorm)
                or tuple(module.normalized_shape) != (vision_hidden,)
                or module.weight is None
                or module.bias is None
            ):
                raise DynamicModelInspectionError(
                    f"SmolVLM vision {label} must be an affine "
                    f"LayerNorm({vision_hidden}) with a bias."
                )

        decoder_graph = self._decoder_graph(
            model,
            model_name=model_name,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
        if dict(decoder_graph.metadata).get("hidden_size") != decoder_hidden:
            raise DynamicModelInspectionError(
                "SmolVLM text_config hidden size differs from the proven "
                "decoder layout."
            )

        module_paths = _smolvlm_module_paths(model)

        def real_path(module: Any, label: str) -> str:
            path = module_paths.get(id(module))
            if path is None:
                raise DynamicModelInspectionError(
                    f"SmolVLM {label} is not a registered child module."
                )
            return path

        view = _SmolVlmTextDecoderView(model)

        def decoder_path(layout_path: str) -> str:
            head, _, rest = layout_path.partition(".")
            if head not in {"model", "lm_head"}:
                raise DynamicModelInspectionError(
                    f"SmolVLM cannot rebase decoder path {layout_path!r}."
                )
            return real_path(
                view.get_submodule(layout_path),
                f"decoder operator {layout_path!r}",
            )

        vision_path = real_path(vision_model, "vision tower")
        connector_path = real_path(connector, "connector")
        composite_path = real_path(composite, "composite model")

        builder = _GraphBuilder(
            model=model,
            model_name=model_name,
            family=SMOLVLM_FAMILY,
            tokenizer_fingerprint=tokenizer_fingerprint,
            groups=_storage_groups(model),
            nodes=[],
            edges=[],
            interfaces=[],
            captured=set(),
        )

        patch_node = builder.add_node(
            "input.image.patch_embedding",
            CircuitKind.MODALITY_INPUT,
            real_path(patch_embedding, "patch embedding"),
            layer_index=None,
            inputs=(
                _port(
                    "pixel_values",
                    PortKind.MODALITY,
                    "image_crops",
                    num_channels,
                    image_size,
                    image_size,
                    modality="image",
                ),
                _port(
                    "patch_attention_mask",
                    PortKind.GENERIC,
                    "image_crops",
                    patches_per_edge,
                    patches_per_edge,
                    modality="image",
                ),
            ),
            outputs=(
                _port(
                    "patches",
                    PortKind.MODALITY,
                    "image_crops",
                    patch_positions,
                    vision_hidden,
                    modality="image",
                ),
            ),
            role_prefix="vision_patch_embedding",
            attributes={
                "input_channels": num_channels,
                "patch_size": patch_size,
                "image_size": image_size,
                "patch_positions": patch_positions,
                "vision_hidden_size": vision_hidden,
                "video_frames_share_this_operator": True,
            },
        )
        position_node = builder.add_node(
            "input.image.position_embedding",
            CircuitKind.POSITION_ENCODING,
            real_path(position_embedding, "image position embedding"),
            layer_index=None,
            inputs=(
                _port(
                    "patches",
                    PortKind.MODALITY,
                    "image_crops",
                    patch_positions,
                    vision_hidden,
                    modality="image",
                ),
            ),
            outputs=(
                _port(
                    "residual",
                    PortKind.RESIDUAL,
                    "image_crops",
                    patch_positions,
                    vision_hidden,
                    modality="image",
                ),
            ),
            role_prefix="vision_position_embedding",
            attributes={
                "kind": "learned_absolute_patch_position",
                "max_positions": patch_positions,
                "vision_hidden_size": vision_hidden,
            },
        )
        builder.edge(
            patch_node,
            "patches",
            position_node,
            "patches",
            EdgeKind.MODALITY,
        )
        builder.interfaces.extend(
            (
                ModalityInterface(
                    interface_id="image.pixels.input",
                    modality="image",
                    direction="input",
                    node_id=patch_node.node_id,
                    port_name="pixel_values",
                    representation=SMOLVLM_IMAGE_REPRESENTATION,
                ),
                ModalityInterface(
                    interface_id="image.patch_mask.input",
                    modality="image",
                    direction="input",
                    node_id=patch_node.node_id,
                    port_name="patch_attention_mask",
                    representation=SMOLVLM_PATCH_MASK_REPRESENTATION,
                ),
            )
        )

        current = position_node
        current_port = "residual"
        for index, layer in enumerate(encoder_layers):
            attention_module = getattr(layer, "self_attn", None)
            attention_norm_module = getattr(layer, "layer_norm1", None)
            mlp_module = getattr(layer, "mlp", None)
            mlp_norm_module = getattr(layer, "layer_norm2", None)
            if any(
                value is None
                for value in (
                    attention_module,
                    attention_norm_module,
                    mlp_module,
                    mlp_norm_module,
                )
            ):
                raise DynamicModelInspectionError(
                    f"SmolVLM vision layer {index} is missing canonical operators."
                )
            require_layer_norm(attention_norm_module, f"layer {index} layer_norm1")
            require_layer_norm(mlp_norm_module, f"layer {index} layer_norm2")
            for name, shape in (
                ("q_proj", (vision_hidden, vision_hidden)),
                ("k_proj", (vision_hidden, vision_hidden)),
                ("v_proj", (vision_hidden, vision_hidden)),
                ("out_proj", (vision_hidden, vision_hidden)),
            ):
                require_linear(
                    getattr(attention_module, name, None),
                    shape,
                    f"layer {index} {name}",
                    bias=True,
                )
            require_linear(
                getattr(mlp_module, "fc1", None),
                (vision_intermediate, vision_hidden),
                f"layer {index} fc1",
                bias=True,
            )
            require_linear(
                getattr(mlp_module, "fc2", None),
                (vision_hidden, vision_intermediate),
                f"layer {index} fc2",
                bias=True,
            )
            if {
                name
                for name, _parameter in attention_module.named_parameters(
                    recurse=True, remove_duplicate=False
                )
            } != {
                "q_proj.weight",
                "q_proj.bias",
                "k_proj.weight",
                "k_proj.bias",
                "v_proj.weight",
                "v_proj.bias",
                "out_proj.weight",
                "out_proj.bias",
            }:
                raise DynamicModelInspectionError(
                    f"SmolVLM vision layer {index} attention owns unsupported "
                    "learned tensors."
                )
            if {
                name
                for name, _parameter in mlp_module.named_parameters(
                    recurse=True, remove_duplicate=False
                )
            } != {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}:
                raise DynamicModelInspectionError(
                    f"SmolVLM vision layer {index} MLP owns unsupported learned "
                    "tensors."
                )

            prefix = f"input.image.encoder.{index:03d}"
            layer_path = real_path(layer, f"vision layer {index}")
            attention_norm = builder.add_node(
                f"{prefix}.attention_norm",
                CircuitKind.NORM,
                real_path(attention_norm_module, f"vision layer {index} norm"),
                layer_index=None,
                inputs=(
                    _port(
                        "residual",
                        PortKind.RESIDUAL,
                        "image_crops",
                        patch_positions,
                        vision_hidden,
                        modality="image",
                    ),
                ),
                outputs=(
                    _port(
                        "normalized",
                        PortKind.NORMALIZED_RESIDUAL,
                        "image_crops",
                        patch_positions,
                        vision_hidden,
                        modality="image",
                    ),
                ),
                role_prefix="vision_attention_norm",
                attributes={
                    "norm_kind": "layer_norm",
                    "epsilon": layer_norm_eps,
                    "weight_kind": "direct_scale",
                    "vision_layer_index": index,
                },
            )
            attention = builder.add_node(
                f"{prefix}.attention",
                CircuitKind.ATTENTION,
                real_path(attention_module, f"vision layer {index} attention"),
                layer_index=None,
                inputs=(
                    _port(
                        "residual",
                        PortKind.NORMALIZED_RESIDUAL,
                        "image_crops",
                        patch_positions,
                        vision_hidden,
                        modality="image",
                    ),
                    _port(
                        "mask",
                        PortKind.ATTENTION_MASK,
                        "image_crops",
                        patch_positions,
                        patch_positions,
                    ),
                ),
                outputs=(
                    _port(
                        "delta",
                        PortKind.BRANCH_DELTA,
                        "image_crops",
                        patch_positions,
                        vision_hidden,
                        modality="image",
                    ),
                ),
                role_prefix="vision_attention",
                attributes={
                    "projection_style": "split_qkv",
                    "query_heads": vision_heads,
                    "key_value_heads": vision_heads,
                    "head_dim": vision_hidden // vision_heads,
                    "grouped_query_attention": False,
                    "layer_type": "bidirectional_attention",
                    "causal": False,
                    "projection_bias": True,
                    "vision_layer_index": index,
                },
            )
            attention_add = builder.add_node(
                f"{prefix}.attention_residual",
                CircuitKind.RESIDUAL_ADD,
                layer_path,
                layer_index=None,
                inputs=(
                    _port(
                        "base",
                        PortKind.RESIDUAL,
                        "image_crops",
                        patch_positions,
                        vision_hidden,
                        modality="image",
                    ),
                    _port(
                        "delta",
                        PortKind.BRANCH_DELTA,
                        "image_crops",
                        patch_positions,
                        vision_hidden,
                        modality="image",
                    ),
                ),
                outputs=(
                    _port(
                        "residual",
                        PortKind.RESIDUAL,
                        "image_crops",
                        patch_positions,
                        vision_hidden,
                        modality="image",
                    ),
                ),
                attributes={
                    "ordered_branch": "attention",
                    "vision_layer_index": index,
                },
            )
            mlp_norm = builder.add_node(
                f"{prefix}.mlp_norm",
                CircuitKind.NORM,
                real_path(mlp_norm_module, f"vision layer {index} mlp norm"),
                layer_index=None,
                inputs=(
                    _port(
                        "residual",
                        PortKind.RESIDUAL,
                        "image_crops",
                        patch_positions,
                        vision_hidden,
                        modality="image",
                    ),
                ),
                outputs=(
                    _port(
                        "normalized",
                        PortKind.NORMALIZED_RESIDUAL,
                        "image_crops",
                        patch_positions,
                        vision_hidden,
                        modality="image",
                    ),
                ),
                role_prefix="vision_mlp_norm",
                attributes={
                    "norm_kind": "layer_norm",
                    "epsilon": layer_norm_eps,
                    "weight_kind": "direct_scale",
                    "vision_layer_index": index,
                },
            )
            mlp = builder.add_node(
                f"{prefix}.mlp",
                CircuitKind.DENSE_MLP,
                real_path(mlp_module, f"vision layer {index} mlp"),
                layer_index=None,
                inputs=(
                    _port(
                        "residual",
                        PortKind.NORMALIZED_RESIDUAL,
                        "image_crops",
                        patch_positions,
                        vision_hidden,
                        modality="image",
                    ),
                ),
                outputs=(
                    _port(
                        "delta",
                        PortKind.BRANCH_DELTA,
                        "image_crops",
                        patch_positions,
                        vision_hidden,
                        modality="image",
                    ),
                ),
                role_prefix="vision_mlp",
                attributes={
                    "intermediate_size": vision_intermediate,
                    "activation": vision_activation,
                    "projection_layout": "output_input",
                    "projection_bias": True,
                    "vision_layer_index": index,
                },
            )
            mlp_add = builder.add_node(
                f"{prefix}.mlp_residual",
                CircuitKind.RESIDUAL_ADD,
                layer_path,
                layer_index=None,
                inputs=(
                    _port(
                        "base",
                        PortKind.RESIDUAL,
                        "image_crops",
                        patch_positions,
                        vision_hidden,
                        modality="image",
                    ),
                    _port(
                        "delta",
                        PortKind.BRANCH_DELTA,
                        "image_crops",
                        patch_positions,
                        vision_hidden,
                        modality="image",
                    ),
                ),
                outputs=(
                    _port(
                        "residual",
                        PortKind.RESIDUAL,
                        "image_crops",
                        patch_positions,
                        vision_hidden,
                        modality="image",
                    ),
                ),
                attributes={
                    "ordered_branch": "mlp",
                    "vision_layer_index": index,
                },
            )
            builder.edge(
                current, current_port, attention_norm, "residual", EdgeKind.DATA
            )
            builder.edge(
                attention_norm, "normalized", attention, "residual", EdgeKind.DATA
            )
            builder.edge(
                current, current_port, attention_add, "base", EdgeKind.RESIDUAL
            )
            builder.edge(attention, "delta", attention_add, "delta", EdgeKind.RESIDUAL)
            builder.edge(attention_add, "residual", mlp_norm, "residual", EdgeKind.DATA)
            builder.edge(mlp_norm, "normalized", mlp, "residual", EdgeKind.DATA)
            builder.edge(attention_add, "residual", mlp_add, "base", EdgeKind.RESIDUAL)
            builder.edge(mlp, "delta", mlp_add, "delta", EdgeKind.RESIDUAL)
            current = mlp_add
            current_port = "residual"

        tower_norm = builder.add_node(
            "input.image.post_layernorm",
            CircuitKind.NORM,
            real_path(post_layernorm, "vision post layernorm"),
            layer_index=None,
            inputs=(
                _port(
                    "residual",
                    PortKind.RESIDUAL,
                    "image_crops",
                    patch_positions,
                    vision_hidden,
                    modality="image",
                ),
            ),
            outputs=(
                _port(
                    "normalized",
                    PortKind.NORMALIZED_RESIDUAL,
                    "image_crops",
                    patch_positions,
                    vision_hidden,
                    modality="image",
                ),
            ),
            role_prefix="vision_post_norm",
            attributes={
                "norm_kind": "layer_norm",
                "epsilon": layer_norm_eps,
                "weight_kind": "direct_scale",
            },
        )
        pixel_shuffle = builder.add_node(
            "input.image.pixel_shuffle",
            CircuitKind.INTERFACE_PROJECTION,
            connector_path,
            layer_index=None,
            inputs=(
                _port(
                    "normalized",
                    PortKind.NORMALIZED_RESIDUAL,
                    "image_crops",
                    patch_positions,
                    vision_hidden,
                    modality="image",
                ),
            ),
            outputs=(
                _port(
                    "shuffled",
                    PortKind.MODALITY,
                    "image_crops",
                    image_tokens_per_crop,
                    shuffled_width,
                    modality="image",
                ),
            ),
            attributes={
                "operation": "spatial_to_channel_pixel_shuffle",
                "scale_factor": scale_factor,
                "image_tokens_per_crop": image_tokens_per_crop,
                "learned_parameters": 0,
            },
        )
        modality_projection_node = builder.add_node(
            "input.image.modality_projection",
            CircuitKind.INTERFACE_PROJECTION,
            real_path(projection, "modality projection"),
            layer_index=None,
            inputs=(
                _port(
                    "shuffled",
                    PortKind.MODALITY,
                    "image_crops",
                    image_tokens_per_crop,
                    shuffled_width,
                    modality="image",
                ),
            ),
            outputs=(
                _port(
                    "image_embeddings",
                    PortKind.MODALITY,
                    "image_crops",
                    image_tokens_per_crop,
                    decoder_hidden,
                    modality="image",
                ),
            ),
            role_prefix="connector_modality_projection",
            attributes={
                "decoder_hidden_size": decoder_hidden,
                "shuffled_vision_width": shuffled_width,
                "projection_bias": False,
            },
        )
        builder.edge(current, current_port, tower_norm, "residual", EdgeKind.DATA)
        builder.edge(
            tower_norm, "normalized", pixel_shuffle, "normalized", EdgeKind.MODALITY
        )
        builder.edge(
            pixel_shuffle,
            "shuffled",
            modality_projection_node,
            "shuffled",
            EdgeKind.MODALITY,
        )

        captured_vision = {
            path
            for path in builder.captured
            if path.startswith(f"{vision_path}.")
            or path.startswith(f"{connector_path}.")
        }
        expected_vision = {
            path
            for path, _parameter in model.named_parameters(
                recurse=True, remove_duplicate=False
            )
            if path.startswith(f"{vision_path}.")
            or path.startswith(f"{connector_path}.")
        }
        if captured_vision != expected_vision:
            missing = sorted(expected_vision - captured_vision)
            extra = sorted(captured_vision - expected_vision)
            raise DynamicModelInspectionError(
                "SmolVLM's vision tower and connector own learned parameters "
                "outside their proven front-end operators; "
                f"unowned={missing[:16]!r}, foreign={extra[:16]!r}."
            )

        decoder_nodes: dict[str, CircuitNode] = {}
        fusion: CircuitNode | None = None
        for node in decoder_graph.nodes:
            roles = tuple(
                parameter.role.split(".", 1)[0] for parameter in node.parameters
            )
            if len(set(roles)) > 1:
                raise DynamicModelInspectionError(
                    "SmolVLM cannot rebase a decoder node that mixes semantic "
                    f"owners: {node.node_id!r}."
                )
            emitted = builder.add_node(
                f"decoder.{node.node_id}",
                node.kind,
                decoder_path(node.path),
                layer_index=node.layer_index,
                inputs=node.inputs,
                outputs=node.outputs,
                role_prefix=roles[0] if roles else None,
                attributes=dict(node.attributes),
            )
            decoder_nodes[node.node_id] = emitted
            if node.kind == CircuitKind.TOKEN_EMBEDDING.value:
                builder.interfaces.append(
                    ModalityInterface(
                        interface_id="text.tokens.input",
                        modality="text",
                        direction="input",
                        node_id=emitted.node_id,
                        port_name="token_ids",
                        representation="token_ids",
                    )
                )
                fusion = builder.add_node(
                    "input.image_text_embedding_fusion",
                    CircuitKind.INTERFACE_PROJECTION,
                    composite_path,
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
                            "image_crops",
                            image_tokens_per_crop,
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
                        "operation": "replace_image_token_embedding_rows",
                        "image_token_id": image_token_id,
                        "image_tokens_per_crop": image_tokens_per_crop,
                        "learned_parameters": 0,
                    },
                )
                builder.edge(
                    emitted, "residual", fusion, "text_residual", EdgeKind.TOKEN
                )
                builder.edge(
                    modality_projection_node,
                    "image_embeddings",
                    fusion,
                    "image_embeddings",
                    EdgeKind.MODALITY,
                )
            if node.kind == CircuitKind.OUTPUT_HEAD.value:
                builder.interfaces.append(
                    ModalityInterface(
                        interface_id="text.logits.output",
                        modality="text",
                        direction="output",
                        node_id=emitted.node_id,
                        port_name="logits",
                        representation="token_logits",
                    )
                )
        if fusion is None:
            raise DynamicModelInspectionError(
                "SmolVLM's decoder exposes no token embedding to fuse image "
                "tokens into."
            )
        for edge in decoder_graph.edges:
            builder.edge(
                decoder_nodes[edge.source_node],
                edge.source_port,
                decoder_nodes[edge.target_node],
                edge.target_port,
                edge.kind,
                recurrent=edge.recurrent,
                attributes=dict(edge.attributes),
            )

        vision_parameters = sum(
            int(parameter.numel())
            for path, parameter in model.named_parameters(
                recurse=True, remove_duplicate=False
            )
            if path.startswith(f"{vision_path}.")
        )
        connector_parameters = sum(
            int(parameter.numel())
            for path, parameter in model.named_parameters(
                recurse=True, remove_duplicate=False
            )
            if path.startswith(f"{connector_path}.")
        )
        decoder_metadata = dict(decoder_graph.metadata)
        return builder.finish(
            metadata={
                "inspector": self.name,
                "decoder_inspector": decoder_metadata["inspector"],
                "decoder_family": decoder_graph.family,
                "decoder_graph_fingerprint": decoder_graph.fingerprint,
                "decoder_config_fingerprint": decoder_graph.config_fingerprint,
                "position_policy": decoder_metadata["position_policy"],
                "layer_count": decoder_metadata["layer_count"],
                "hidden_size": decoder_hidden,
                "tie_word_embeddings": decoder_metadata["tie_word_embeddings"],
                "vision_hidden_size": vision_hidden,
                "vision_layer_count": vision_layers,
                "vision_parameter_count": vision_parameters,
                "connector_parameter_count": connector_parameters,
                "image_tokens_per_crop": image_tokens_per_crop,
                "image_token_id": image_token_id,
                "pixel_shuffle_scale_factor": scale_factor,
                "native_rgb_crop_path": True,
                "image_token_replacement_fusion": True,
                "video_lane": SMOLVLM_VIDEO_NOTE,
            }
        )


# ---------------------------------------------------------------------------
# chimera-omni: encoder-free native omnimodal decoder trunk.
#
# chimera-omni is not a Hugging Face model and is not meant to become one.  It
# is a plain ``torch.nn.Module`` whose frozen configuration lives at ``cfg``,
# whose media path is one shared boundary projection instead of a tower per
# modality, and whose depth axis carries a learned selection over checkpointed
# residual prefix sums (``residual_mode="block_attnres"``).  Everything below
# compiles that topology into the canonical IR without importing, vendoring or
# copying one line of the target repository: the contract is proven from the
# live module tree and the configuration manifest the model emits itself.
# ---------------------------------------------------------------------------


CHIMERA_OMNI_FAMILY = "chimera_omni"
CHIMERA_OMNI_CONFIG_SCHEMA = "chimera_omni.config.v1"
CHIMERA_OMNI_ARCHITECTURES = frozenset(
    {
        "dense_decoder_native_omnimodal_encoder_free",
        "stable_latent_moe_decoder_native_omnimodal_encoder_free",
    }
)
CHIMERA_OMNI_BLOCK_KINDS = ("kda", "local", "csa", "hca", "gated_mla")
CHIMERA_OMNI_FEED_FORWARD_KINDS = ("dense", "stable_latent_moe")
CHIMERA_OMNI_RESIDUAL_MODES = ("none", "block_attnres")
CHIMERA_OMNI_POSITION_POLICY = "nope"
CHIMERA_OMNI_MEDIA_TOKEN_DIMENSION = "native_media_tokens"
CHIMERA_OMNI_TEXT_MODALITY = canonical_modality("text", "chimera-omni text modality")

# ``chimera_omni.native_segments.NATIVE_SEGMENT_SPECS`` and
# ``chimera_omni.native_media.MEDIA_TYPE_IDS`` declare six native media lanes
# and their trunk type ids.  ``music`` and ``tts`` are not canonical IR
# modality names, and they are not new modalities either: both are raw waveform
# segment lanes that ``waveform_to_native_segments`` builds exactly the way it
# builds ``audio``.  They therefore enter the IR as canonical ``audio`` while
# the chimera lane name stays on the interface id and the representation, and
# the mapping is stated in graph metadata rather than applied silently.
CHIMERA_OMNI_MEDIA_LANES: tuple[tuple[str, int, str], ...] = (
    ("image", 1, canonical_modality("image", "chimera-omni image lane")),
    ("video", 2, canonical_modality("video", "chimera-omni video lane")),
    ("audio", 3, canonical_modality("audio", "chimera-omni audio lane")),
    ("music", 4, canonical_modality("audio", "chimera-omni music lane")),
    ("tts", 5, canonical_modality("audio", "chimera-omni tts lane")),
    ("ocr", 6, canonical_modality("ocr", "chimera-omni ocr lane")),
)
CHIMERA_OMNI_MAX_MEDIA_TYPE_ID = max(
    type_id for _lane, type_id, _canonical in CHIMERA_OMNI_MEDIA_LANES
)
CHIMERA_OMNI_LANE_CANONICALIZATION = (
    "image->image, video->video, audio->audio, music->audio, tts->audio, ocr->ocr"
)

# The learned tensor set each mixer family must own, exactly.  A block that
# owns more or fewer tensors than its declared family predicts is refused by
# name rather than absorbed into a vaguer node.
_CHIMERA_LOCAL_ATTENTION_PARAMETERS = frozenset(
    {"q_proj.weight", "k_proj.weight", "v_proj.weight", "out.weight"}
)
_CHIMERA_SPARSE_LATENT_PARAMETERS = frozenset(
    {
        "local.q_proj.weight",
        "local.k_proj.weight",
        "local.v_proj.weight",
        "local.out.weight",
        "summary_q.weight",
        "summary_k.weight",
        "summary_v.weight",
        "out.weight",
    }
)
_CHIMERA_KDA_PARAMETERS = frozenset(
    {
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "q_conv.conv.weight",
        "q_conv.conv.bias",
        "k_conv.conv.weight",
        "k_conv.conv.bias",
        "v_conv.conv.weight",
        "v_conv.conv.bias",
        "decay_down.weight",
        "decay_up.weight",
        "decay_bias",
        "decay_log_scale",
        "write_strength.weight",
        "output_gate.weight",
        "output_norm.weight",
        "out.weight",
    }
)
_CHIMERA_GATED_MLA_LATENT_QUERY_PARAMETERS = frozenset(
    {
        "q_down.weight",
        "q_norm.weight",
        "q_up.weight",
        "kv_down.weight",
        "kv_norm.weight",
        "kv_up.weight",
        "gate.weight",
        "out.weight",
    }
)
_CHIMERA_GATED_MLA_DENSE_QUERY_PARAMETERS = frozenset(
    {
        "q_proj.weight",
        "kv_down.weight",
        "kv_norm.weight",
        "kv_up.weight",
        "gate.weight",
        "out.weight",
    }
)
_CHIMERA_DENSE_FEED_FORWARD_PARAMETERS = frozenset({"up.weight", "down.weight"})

# Which fold-kernel family each mixer needs.  The inspector never decides
# whether a fold may run -- that is the planner's and the operator registry's
# job -- but naming the required kernel family on the node is what makes their
# refusal legible instead of mysterious.
_CHIMERA_MIXER_KERNEL_FAMILIES: Mapping[str, str] = {
    "local": "causal_sliding_window_split_qkv_attention",
    "csa": "compressed_summary_attention",
    "hca": "compressed_summary_attention",
    "kda": "bounded_gated_delta_recurrence",
    "gated_mla": "compressed_latent_attention",
}
# Families whose IR decomposition matches the shape a promoted causal branch
# operator in this repository is written against: one pre-operator norm, one
# complete coupled operator node, one ordered residual write, and a registered
# fold kernel for the declared family.  ``local`` folds through the windowed
# NoPE GQA kernel, ``csa``/``hca`` through the stride-parameterized sparse
# latent kernel, ``kda`` through the bounded gated delta-rule kernel, and
# ``gated_mla`` through the compressed-latent-KV kernel
# (``docs/GATED_MLA_MOE_FOLD_KERNEL_DESIGN_2026-08-05.md`` section 1).  Every
# declared chimera sequence-mixer family is now covered.
_CHIMERA_STRUCTURALLY_COVERABLE_MIXERS = frozenset(
    {"local", "csa", "hca", "kda", "gated_mla"}
)
#: The fold-kernel family the Stable LatentMoE channel mixer needs.  It is one
#: name, not a router name plus an expert name, because the module makes one
#: ordered residual write and is therefore one composite operator.
_CHIMERA_STABLE_LATENT_MOE_KERNEL_FAMILY = "stable_latent_moe_channel_mixer"


class ChimeraOmniInspectionError(DynamicModelInspectionError):
    """One chimera-omni topology cannot be proven by this inspector."""


class _ChimeraOmniConfigDocument:
    """The chimera configuration manifest presented as a config object."""

    __slots__ = ("_payload",)

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._payload = dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload)


class _ChimeraOmniLayoutView:
    """The chimera model with its configuration under the shared name.

    ``ChimeraOmniModel`` keeps its frozen ``ChimeraConfig`` at ``cfg`` and
    exposes no ``config`` attribute, so :func:`_config_payload` cannot seal it.
    This view renames exactly that one attribute against the *same* module
    object -- nothing is copied, nothing is re-registered, and every parameter
    path, module path and submodule it reports is the model's own -- so the
    shared graph builder fingerprints the chimera configuration the way it
    fingerprints every other family's.  The finished graph is re-emitted under
    the real model class, so this view never appears in the sealed IR.
    """

    __slots__ = ("_model", "config")

    def __init__(self, model: Any, payload: Mapping[str, Any]) -> None:
        self._model = model
        self.config = _ChimeraOmniConfigDocument(payload)

    def named_parameters(
        self,
        recurse: bool = True,
        remove_duplicate: bool = False,
    ) -> Any:
        return self._model.named_parameters(
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )

    def named_modules(self) -> Any:
        return self._model.named_modules()

    def get_submodule(self, path: str) -> Any:
        return self._model.get_submodule(path)

    def parameters(self) -> Any:
        return self._model.parameters()


def _chimera_graph_with_model_class(
    graph: ModelCircuitGraph,
    model_class: str,
) -> ModelCircuitGraph:
    """Re-emit one finished graph under the wrapped model's real class name."""

    from dataclasses import replace

    return replace(graph, model_class=model_class)


class ChimeraOmniCircuitInspector:
    """Compile a chimera-omni encoder-free native omnimodal decoder trunk.

    Three properties separate this family from every lane already compiled
    here, and all three are represented explicitly rather than flattened away:

    * **The media path has no tower.**  One ``NativeContinuousMediaBridge``
      pushes the continuous boundary patches of *every* modality through a
      single learned ``feature_proj``, adds one typed-ledger type embedding and
      one continuous position projection, and scatters the result into the
      trunk residual stream at the caller's media token positions.  The
      ``MODALITY_INPUT`` decomposition therefore carries one typed port per
      chimera media lane while owning exactly one projection tensor: the port
      set states the multi-modality entry, the parameter set states that all of
      them share it.
    * **The model emits media.**  ``UnifiedNativeMediaEmissionHead`` is the
      second real ``CircuitKind.MODALITY_OUTPUT`` emitter in this repository,
      after the points3d lane's quantized centroid head.  It reads the same
      final-normed hidden rows the text head reads, so the two heads are
      siblings rather than a chain, and it exposes one ``direction='output'``
      interface per media lane.
    * **The depth axis is learned.**  Under ``residual_mode='block_attnres'``
      the trunk checkpoints residual prefix sums and mixes them with a learned
      pseudo-query before each branch reads them.  Every depth source is an
      explicit ``EdgeKind.RESIDUAL`` edge into a ``CircuitKind.RESIDUAL_ADD``
      node that owns the selection tensors, so the depth topology lives in the
      graph instead of in a comment.

    Mixer families are described honestly and refused by name.  ``local`` is
    causal sliding-window split-Q/K/V/O grouped-query attention; ``csa`` and
    ``hca`` are one composite ``ATTENTION`` operator that runs a local window
    and a strided compressed summary over the *same* normalized rows and makes
    exactly one residual write, so they are emitted as one node whose two
    sub-branches are declared attributes rather than as two nodes joined by a
    residual add the model never performs; ``gated_mla`` is compressed-latent
    attention with a full-rank output gate; ``kda`` is a
    bounded gated delta-rule recurrence carrying an explicit
    ``[batch, heads, key_dim, value_dim]`` state, which is what
    ``CircuitKind.RECURRENT_MIXER`` means, so it is emitted as one -- with the
    fold-kernel family it still needs named on the node.  Any other family, or
    any family whose live module owns a tensor set its declaration does not
    predict, fails closed naming the family.
    """

    name = "chimera-omni-native-decoder"
    priority = 245

    # -- selection ----------------------------------------------------------

    @staticmethod
    def _config(model: Any) -> Any:
        config = getattr(model, "cfg", None)
        if config is None:
            config = getattr(model, "config", None)
        return config

    def supports(self, model: Any) -> bool:
        config = self._config(model)
        return bool(
            config is not None
            and callable(getattr(config, "to_manifest", None))
            and getattr(config, "block_pattern", None) is not None
            and getattr(config, "residual_mode", None) is not None
            and getattr(config, "native_feature_dim", None) is not None
            and hasattr(model, "embed")
            and hasattr(model, "blocks")
            and hasattr(model, "lm_head")
            and hasattr(model, "native_media_bridge")
            and hasattr(model, "native_media_head")
        )

    # -- configuration ------------------------------------------------------

    def _manifest(self, config: Any) -> dict[str, Any]:
        to_manifest = getattr(config, "to_manifest", None)
        if not callable(to_manifest):
            raise ChimeraOmniInspectionError(
                "chimera-omni inspection requires a ChimeraConfig that emits its "
                "own manifest."
            )
        try:
            payload = to_manifest()
        except Exception as exc:  # noqa: BLE001 - the target repo owns this error
            raise ChimeraOmniInspectionError(
                f"chimera-omni configuration refused to validate: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ChimeraOmniInspectionError(
                "chimera-omni configuration manifest is not a mapping."
            )
        manifest = dict(payload)
        if manifest.get("schema") != CHIMERA_OMNI_CONFIG_SCHEMA:
            raise ChimeraOmniInspectionError(
                "chimera-omni inspection requires a "
                f"{CHIMERA_OMNI_CONFIG_SCHEMA} manifest; got "
                f"{manifest.get('schema')!r}."
            )
        architecture = manifest.get("architecture")
        if architecture not in CHIMERA_OMNI_ARCHITECTURES:
            raise ChimeraOmniInspectionError(
                f"chimera-omni architecture {architecture!r} is not one of the "
                "encoder-free native omnimodal decoder architectures: "
                + ", ".join(sorted(CHIMERA_OMNI_ARCHITECTURES))
                + "."
            )
        return manifest

    def _geometry(
        self,
        manifest: Mapping[str, Any],
        config: Any,
    ) -> dict[str, Any]:
        def integer(name: str, *, minimum: int = 1) -> int:
            value = manifest.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ChimeraOmniInspectionError(
                    f"chimera-omni {name} must be an integer of at least "
                    f"{minimum}; got {value!r}."
                )
            return int(value)

        def runtime_integer(name: str, *, minimum: int = 0) -> int:
            value = getattr(config, name, None)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ChimeraOmniInspectionError(
                    f"chimera-omni config.{name} must be an integer of at least "
                    f"{minimum}; got {value!r}."
                )
            return int(value)

        hidden = integer("hidden_size")
        heads = integer("num_heads")
        key_value_heads = integer("num_kv_heads")
        if heads % key_value_heads:
            raise ChimeraOmniInspectionError(
                "chimera-omni num_heads must be divisible by num_kv_heads."
            )
        declared_head_dim = integer("attention_head_dim", minimum=0)
        if declared_head_dim:
            head_dim = declared_head_dim
        elif hidden % heads:
            raise ChimeraOmniInspectionError(
                "chimera-omni needs an explicit attention_head_dim when "
                "hidden_size is not divisible by num_heads."
            )
        else:
            head_dim = hidden // heads
        residual_mode = manifest.get("residual_mode")
        if residual_mode not in CHIMERA_OMNI_RESIDUAL_MODES:
            raise ChimeraOmniInspectionError(
                f"chimera-omni residual_mode {residual_mode!r} is not one of "
                + ", ".join(CHIMERA_OMNI_RESIDUAL_MODES)
                + "."
            )
        feed_forward_kind = manifest.get("feed_forward_kind")
        if feed_forward_kind not in CHIMERA_OMNI_FEED_FORWARD_KINDS:
            raise ChimeraOmniInspectionError(
                f"chimera-omni channel mixer {feed_forward_kind!r} has no "
                "inspected decomposition; the supported channel mixers are "
                + ", ".join(CHIMERA_OMNI_FEED_FORWARD_KINDS)
                + "."
            )
        native_type_count = integer("native_type_count")
        if native_type_count <= CHIMERA_OMNI_MAX_MEDIA_TYPE_ID:
            raise ChimeraOmniInspectionError(
                f"chimera-omni native_type_count {native_type_count} cannot "
                "address the native media type ids the segment manifest "
                f"declares (highest is {CHIMERA_OMNI_MAX_MEDIA_TYPE_ID})."
            )
        block_pattern = manifest.get("block_pattern")
        if not isinstance(block_pattern, Sequence) or isinstance(block_pattern, str):
            raise ChimeraOmniInspectionError(
                "chimera-omni block_pattern must be a sequence of block families."
            )
        pattern = tuple(str(value) for value in block_pattern)
        if not pattern:
            raise ChimeraOmniInspectionError("chimera-omni block_pattern is empty.")
        for index, kind in enumerate(pattern):
            if kind not in CHIMERA_OMNI_BLOCK_KINDS:
                raise ChimeraOmniInspectionError(
                    f"chimera-omni sequence-mixing family {kind!r} at "
                    f"block_pattern index {index} has no inspected "
                    "decomposition; the supported families are "
                    + ", ".join(CHIMERA_OMNI_BLOCK_KINDS)
                    + "."
                )
        aux_heads = manifest.get("aux_heads")
        if not isinstance(aux_heads, Sequence) or isinstance(aux_heads, str):
            raise ChimeraOmniInspectionError(
                "chimera-omni aux_heads must be a sequence of head names."
            )
        gate_lower_bound = manifest.get("kda_gate_lower_bound")
        if not isinstance(gate_lower_bound, (int, float)) or isinstance(
            gate_lower_bound, bool
        ):
            raise ChimeraOmniInspectionError(
                "chimera-omni kda_gate_lower_bound must be a real number."
            )
        key_head_dim = integer("kda_head_dim", minimum=0) or head_dim
        mla_key_head_dim = integer("mla_key_head_dim", minimum=0) or head_dim
        return {
            "vocab_size": integer("vocab_size"),
            "hidden_size": hidden,
            "intermediate_size": integer("intermediate_size"),
            "num_layers": integer("num_layers"),
            "num_heads": heads,
            "num_kv_heads": key_value_heads,
            "head_dim": head_dim,
            "declared_head_dim": declared_head_dim,
            "kda_key_head_dim": key_head_dim,
            "kda_value_head_dim": (
                integer("kda_value_head_dim", minimum=0) or key_head_dim
            ),
            "kda_decay_rank": integer("kda_decay_rank", minimum=0) or key_head_dim,
            "kda_short_conv_kernel_size": integer("kda_short_conv_kernel_size"),
            "kda_gate_lower_bound": float(gate_lower_bound),
            "mla_key_head_dim": mla_key_head_dim,
            "mla_value_head_dim": (
                integer("mla_value_head_dim", minimum=0) or mla_key_head_dim
            ),
            "mla_extra_head_dim": integer("mla_extra_head_dim", minimum=0),
            "mla_latent_size": (
                integer("mla_latent_size", minimum=0)
                or max(mla_key_head_dim, hidden // 2)
            ),
            "mla_query_latent_size": integer("mla_query_latent_size", minimum=0),
            "moe_num_experts": integer("moe_num_experts", minimum=0),
            "moe_experts_per_token": integer("moe_experts_per_token", minimum=0),
            "moe_shared_experts": integer("moe_shared_experts", minimum=0),
            "moe_latent_size": integer("moe_latent_size", minimum=0),
            "moe_first_dense_layers": integer("moe_first_dense_layers", minimum=0),
            "moe_routed_intermediate_size": integer(
                "moe_routed_intermediate_size", minimum=0
            ),
            "native_feature_dim": integer("native_feature_dim"),
            "native_position_dim": integer("native_position_dim"),
            "native_type_count": native_type_count,
            "mtp_heads": integer("mtp_heads", minimum=0),
            "attnres_block_size": integer("attnres_block_size", minimum=0),
            "force_final_gated_mla": bool(manifest.get("force_final_gated_mla")),
            "residual_mode": str(residual_mode),
            "feed_forward_kind": str(feed_forward_kind),
            "activation_kind": str(manifest.get("activation_kind", "")),
            # The SiTU caps are real arithmetic of the gated activation, so a
            # channel mixer that runs them declares them rather than leaving a
            # kernel to guess the two betas.
            "situ_gate_beta": float(manifest.get("situ_gate_beta", 0.0) or 0.0),
            "situ_up_beta": float(manifest.get("situ_up_beta", 0.0) or 0.0),
            "block_pattern": pattern,
            "aux_heads": tuple(str(value) for value in aux_heads),
            # Window and compression strides are real configuration that the
            # chimera manifest does not carry, so they are read from the live
            # config object and recorded as such.
            "local_window": runtime_integer("local_window"),
            "csa_compress_stride": runtime_integer("csa_compress_stride"),
            "hca_compress_stride": runtime_integer("hca_compress_stride"),
        }

    # -- node families ------------------------------------------------------

    def _depth_select_node(
        self,
        builder: _GraphBuilder,
        *,
        node_id: str,
        path: str,
        role_prefix: str,
        layer_index: int | None,
        hidden: int,
        source_count: int,
        residual_port: Any,
        attributes: Mapping[str, Any],
    ) -> CircuitNode:
        """One learned depth-axis selection over checkpointed prefix sums.

        ``DepthAttentionResidual`` scores ``[*completed_blocks, prefix_sum]``
        with a learned pseudo-query over RMS-normalized depth sources and mixes
        them by softmax.  Its output is what the following branch *reads*, while
        that branch still *writes* into the running prefix sum, so this is a
        residual-stream node with one explicit ``EdgeKind.RESIDUAL`` input per
        depth source rather than an ordinary two-term add.  At a depth where no
        block has completed yet the module is never invoked and the residual
        passes through unchanged; that is recorded rather than hidden, because
        those tensors are then dead weight at that depth.
        """

        inputs = [residual_port("residual")]
        inputs.extend(
            residual_port(f"depth_source_{slot:03d}") for slot in range(source_count)
        )
        return builder.add_node(
            node_id,
            CircuitKind.RESIDUAL_ADD,
            path,
            layer_index=layer_index,
            inputs=tuple(inputs),
            outputs=(residual_port("selected"),),
            role_prefix=role_prefix,
            attributes={
                "residual_mode": "block_attnres",
                "depth_source_count": source_count,
                "mixed_source_count": source_count + 1,
                "selection": "learned_softmax_over_rms_normalized_depth_sources",
                "learned_pseudo_query": True,
                "invoked_at_this_depth": source_count > 0,
                "identity_passthrough": source_count == 0,
                "hidden_size": hidden,
                **dict(attributes),
            },
        )

    def _mixer_nodes(
        self,
        builder: _GraphBuilder,
        *,
        model: Any,
        prefix: str,
        block_path: str,
        block_kind: str,
        layer_index: int,
        geometry: Mapping[str, Any],
        normalized_port: Any,
        delta_port: Any,
        require_linear: Any,
        require_rms_norm: Any,
        require_parameter_set: Any,
        operator_norm: CircuitNode,
    ) -> tuple[CircuitNode, str]:
        """Compile one sequence mixer, refusing unknown families by name."""

        hidden = int(geometry["hidden_size"])
        heads = int(geometry["num_heads"])
        key_value_heads = int(geometry["num_kv_heads"])
        head_dim = int(geometry["head_dim"])
        local_window = int(geometry["local_window"])
        mixer_path = f"{block_path}.mixer"
        kernel_family = _CHIMERA_MIXER_KERNEL_FAMILIES[block_kind]

        def attention_ports() -> tuple[
            tuple[CircuitPort, ...],
            tuple[CircuitPort, ...],
        ]:
            return (
                (
                    normalized_port("residual"),
                    _port(
                        "mask",
                        PortKind.ATTENTION_MASK,
                        "batch",
                        "sequence",
                        "sequence",
                    ),
                    _port("cache_in", PortKind.CACHE, "layers", stateful=True),
                ),
                (
                    delta_port("delta"),
                    _port("cache_out", PortKind.CACHE, "layers", stateful=True),
                ),
            )

        def require_local_attention(path: str, label: str) -> None:
            require_linear(
                f"{path}.q_proj", (heads * head_dim, hidden), f"{label} q_proj"
            )
            require_linear(
                f"{path}.k_proj",
                (key_value_heads * head_dim, hidden),
                f"{label} k_proj",
            )
            require_linear(
                f"{path}.v_proj",
                (key_value_heads * head_dim, hidden),
                f"{label} v_proj",
            )
            require_linear(f"{path}.out", (hidden, heads * head_dim), f"{label} out")

        def local_attributes() -> dict[str, Any]:
            return {
                "projection_style": "split_qkv",
                "query_heads": heads,
                "key_value_heads": key_value_heads,
                "head_dim": head_dim,
                "grouped_query_attention": heads != key_value_heads,
                "gqa_group": heads // key_value_heads,
                "layer_type": (
                    "sliding_window_attention" if local_window > 0 else "full_attention"
                ),
                "sliding_window": local_window if local_window > 0 else None,
                "causal": True,
                "projection_bias": False,
                "position_policy": CHIMERA_OMNI_POSITION_POLICY,
                "operator_type": "local_causal_attention",
                "fold_kernel_family_required": _CHIMERA_MIXER_KERNEL_FAMILIES["local"],
            }

        if block_kind == "local":
            require_parameter_set(
                mixer_path,
                _CHIMERA_LOCAL_ATTENTION_PARAMETERS,
                "local mixer",
                "local",
            )
            require_local_attention(mixer_path, f"layer {layer_index} local attention")
            inputs, outputs = attention_ports()
            node = builder.add_node(
                f"{prefix}.operator",
                CircuitKind.ATTENTION,
                mixer_path,
                layer_index=layer_index,
                inputs=inputs,
                outputs=outputs,
                role_prefix="operator",
                attributes=local_attributes(),
            )
            builder.edge(operator_norm, "normalized", node, "residual", EdgeKind.DATA)
            builder.edge(
                node, "cache_out", node, "cache_in", EdgeKind.CACHE, recurrent=True
            )
            return node, "delta"

        if block_kind in {"csa", "hca"}:
            require_parameter_set(
                mixer_path,
                _CHIMERA_SPARSE_LATENT_PARAMETERS,
                "compressed-summary mixer",
                block_kind,
            )
            require_local_attention(
                f"{mixer_path}.local",
                f"layer {layer_index} {block_kind} local window",
            )
            for name in ("summary_q", "summary_k", "summary_v", "out"):
                require_linear(
                    f"{mixer_path}.{name}",
                    (hidden, hidden),
                    f"layer {layer_index} {block_kind} {name}",
                )
            stride = int(
                geometry[
                    "csa_compress_stride" if block_kind == "csa" else (
                        "hca_compress_stride"
                    )
                ]
            )
            inputs, outputs = attention_ports()
            # One composite operator, not a local node plus a summary node.
            # ``SparseLatentAttention.forward`` computes both sub-branches from
            # the same normalized rows and returns ``local + out(summary)`` --
            # one value, written into the residual stream exactly once.  Two
            # nodes joined by a RESIDUAL_ADD would encode a residual write the
            # model does not perform and would break the family-neutral
            # one-operator-per-branch contract
            # (``docs/KDA_CSA_HCA_FOLD_KERNEL_DESIGN_2026-08-04.md`` section 4,
            # ``NOTES-W11B.md``).  The sub-branch structure survives as declared
            # attributes, which is what it is: internal structure of one
            # operator.
            node = builder.add_node(
                f"{prefix}.operator",
                CircuitKind.ATTENTION,
                mixer_path,
                layer_index=layer_index,
                inputs=inputs,
                outputs=outputs,
                role_prefix="operator",
                attributes={
                    **local_attributes(),
                    "projection_style": (
                        "split_qkv_local_window_plus_full_width_compressed_summary"
                    ),
                    "layer_type": "compressed_summary_attention",
                    "operator_type": "sparse_latent_attention",
                    "composite_operator": True,
                    "composite_sub_branches": "local_window,compressed_summary",
                    "composite_residual_writes": 1,
                    "sub_branch_combination": (
                        "sum_of_local_window_and_compressed_summary"
                    ),
                    "local_window": local_window if local_window > 0 else None,
                    "summary_stride": stride,
                    "summary_heads": 1,
                    "summary_head_dim": hidden,
                    "summary_projection_style": "split_qkv_full_width",
                    "summary_reduction": "mean_over_stride_with_leading_zero_summary",
                    "mixer_family": block_kind,
                    "fold_kernel_family_required": kernel_family,
                },
            )
            builder.edge(operator_norm, "normalized", node, "residual", EdgeKind.DATA)
            builder.edge(
                node, "cache_out", node, "cache_in", EdgeKind.CACHE, recurrent=True
            )
            return node, "delta"

        if block_kind == "kda":
            require_parameter_set(
                mixer_path, _CHIMERA_KDA_PARAMETERS, "KDA mixer", "kda"
            )
            key_head_dim = int(geometry["kda_key_head_dim"])
            value_head_dim = int(geometry["kda_value_head_dim"])
            decay_rank = int(geometry["kda_decay_rank"])
            kernel_size = int(geometry["kda_short_conv_kernel_size"])
            key_width = heads * key_head_dim
            value_width = heads * value_head_dim
            require_linear(f"{mixer_path}.q_proj", (key_width, hidden), "KDA q_proj")
            require_linear(f"{mixer_path}.k_proj", (key_width, hidden), "KDA k_proj")
            require_linear(f"{mixer_path}.v_proj", (value_width, hidden), "KDA v_proj")
            require_linear(
                f"{mixer_path}.decay_down", (decay_rank, hidden), "KDA decay_down"
            )
            require_linear(
                f"{mixer_path}.decay_up", (key_width, decay_rank), "KDA decay_up"
            )
            require_linear(
                f"{mixer_path}.write_strength", (heads, hidden), "KDA write_strength"
            )
            require_linear(
                f"{mixer_path}.output_gate", (value_width, hidden), "KDA output_gate"
            )
            require_linear(f"{mixer_path}.out", (hidden, value_width), "KDA out")
            for name, width in (
                ("q_conv", key_width),
                ("k_conv", key_width),
                ("v_conv", value_width),
            ):
                convolution = model.get_submodule(f"{mixer_path}.{name}.conv")
                if (
                    tuple(int(value) for value in convolution.weight.shape)
                    != (width, 1, kernel_size)
                    or convolution.bias is None
                    or int(convolution.groups) != width
                ):
                    raise ChimeraOmniInspectionError(
                        f"chimera-omni KDA {name} at layer {layer_index} must be a "
                        f"depthwise causal Conv1d({width}, {width}, "
                        f"kernel_size={kernel_size}) with a bias."
                    )
            node = builder.add_node(
                f"{prefix}.operator",
                CircuitKind.RECURRENT_MIXER,
                mixer_path,
                layer_index=layer_index,
                inputs=(
                    normalized_port("residual"),
                    _port(
                        "state_in", PortKind.RECURRENT_STATE, "layers", stateful=True
                    ),
                ),
                outputs=(
                    delta_port("delta"),
                    _port(
                        "state_out", PortKind.RECURRENT_STATE, "layers", stateful=True
                    ),
                ),
                role_prefix="operator",
                attributes={
                    "operator_type": "kimi_delta_attention",
                    "recurrence": "bounded_gated_delta_rule",
                    "heads": heads,
                    "key_head_dim": key_head_dim,
                    "value_head_dim": value_head_dim,
                    "decay_rank": decay_rank,
                    "gate_lower_bound": float(geometry["kda_gate_lower_bound"]),
                    "state_elements": heads * key_head_dim * value_head_dim,
                    "state_layout": "batch_heads_key_value",
                    "internal_short_conv_operators": 3,
                    "internal_short_conv_kernel_size": kernel_size,
                    "internal_short_conv_paths": "q_conv,k_conv,v_conv",
                    "internal_headwise_output_norm": True,
                    "not_attention_qkv": True,
                    "softmax_free": True,
                    "position_policy": CHIMERA_OMNI_POSITION_POLICY,
                    "fold_kernel_family_required": kernel_family,
                },
            )
            builder.edge(operator_norm, "normalized", node, "residual", EdgeKind.DATA)
            builder.edge(
                node, "state_out", node, "state_in", EdgeKind.STATE, recurrent=True
            )
            return node, "delta"

        if block_kind == "gated_mla":
            query_latent = int(geometry["mla_query_latent_size"])
            require_parameter_set(
                mixer_path,
                (
                    _CHIMERA_GATED_MLA_LATENT_QUERY_PARAMETERS
                    if query_latent > 0
                    else _CHIMERA_GATED_MLA_DENSE_QUERY_PARAMETERS
                ),
                "gated MLA mixer",
                "gated_mla",
            )
            key_head_dim = int(geometry["mla_key_head_dim"])
            value_head_dim = int(geometry["mla_value_head_dim"])
            extra_head_dim = int(geometry["mla_extra_head_dim"])
            latent = int(geometry["mla_latent_size"])
            query_head_dim = key_head_dim + extra_head_dim
            if query_latent > 0:
                require_linear(
                    f"{mixer_path}.q_down", (query_latent, hidden), "MLA q_down"
                )
                require_rms_norm(f"{mixer_path}.q_norm", query_latent, "MLA q_norm")
                require_linear(
                    f"{mixer_path}.q_up",
                    (heads * query_head_dim, query_latent),
                    "MLA q_up",
                )
            else:
                require_linear(
                    f"{mixer_path}.q_proj",
                    (heads * query_head_dim, hidden),
                    "MLA q_proj",
                )
            require_linear(
                f"{mixer_path}.kv_down",
                (latent + extra_head_dim, hidden),
                "MLA kv_down",
            )
            require_rms_norm(f"{mixer_path}.kv_norm", latent, "MLA kv_norm")
            require_linear(
                f"{mixer_path}.kv_up",
                (heads * (key_head_dim + value_head_dim), latent),
                "MLA kv_up",
            )
            require_linear(
                f"{mixer_path}.gate", (heads * value_head_dim, hidden), "MLA gate"
            )
            require_linear(
                f"{mixer_path}.out", (hidden, heads * value_head_dim), "MLA out"
            )
            inputs, outputs = attention_ports()
            node = builder.add_node(
                f"{prefix}.operator",
                CircuitKind.ATTENTION,
                mixer_path,
                layer_index=layer_index,
                inputs=inputs,
                outputs=outputs,
                role_prefix="operator",
                attributes={
                    "projection_style": "compressed_latent_kv",
                    "query_heads": heads,
                    "key_value_heads": heads,
                    "head_dim": query_head_dim,
                    "key_head_dim": key_head_dim,
                    "value_head_dim": value_head_dim,
                    "extra_head_dim": extra_head_dim,
                    "kv_latent_size": latent,
                    "query_latent_size": query_latent,
                    "grouped_query_attention": False,
                    "layer_type": "full_attention",
                    "causal": True,
                    "full_rank_output_gate": True,
                    "projection_bias": False,
                    "position_policy": CHIMERA_OMNI_POSITION_POLICY,
                    "operator_type": "gated_multi_head_latent_attention",
                    "fold_kernel_family_required": kernel_family,
                },
            )
            builder.edge(operator_norm, "normalized", node, "residual", EdgeKind.DATA)
            builder.edge(
                node, "cache_out", node, "cache_in", EdgeKind.CACHE, recurrent=True
            )
            return node, "delta"

        raise ChimeraOmniInspectionError(
            f"chimera-omni sequence-mixing family {block_kind!r} has no inspected "
            "decomposition; the supported families are "
            + ", ".join(CHIMERA_OMNI_BLOCK_KINDS)
            + "."
        )

    def _channel_mixer_nodes(
        self,
        builder: _GraphBuilder,
        *,
        prefix: str,
        block_path: str,
        layer_index: int,
        geometry: Mapping[str, Any],
        normalized_port: Any,
        delta_port: Any,
        require_linear: Any,
        require_rms_norm: Any,
        require_parameter_set: Any,
        feed_forward_norm: CircuitNode,
    ) -> tuple[CircuitNode, str]:
        """Compile one channel mixer: fused-gate dense SwiGLU or Stable LatentMoE."""

        hidden = int(geometry["hidden_size"])
        intermediate = int(geometry["intermediate_size"])
        activation = str(geometry["activation_kind"])
        path = f"{block_path}.ffn"
        dense = geometry["feed_forward_kind"] != "stable_latent_moe" or (
            layer_index < int(geometry["moe_first_dense_layers"])
        )
        full_width_attributes = {
            "intermediate_size": intermediate,
            "activation": activation,
            "projection_layout": "fused_gate_up_then_down",
            "fused_gate_up_projection": True,
            "gate_up_split": "chunk_two_on_last_dimension",
            "projection_bias": False,
            "fold_kernel_family_required": "fused_gate_up_swiglu",
        }
        # SiTU caps *both* halves of the gate/up split with its own betas
        # before gating (``layers.py:350-355``), so they are part of the dense
        # mixer's arithmetic and the planner has to check them for equality.
        # SiLU reads neither, so a SiLU mixer declares neither: a constant no
        # arithmetic consumes is not a contract, and emitting it would move
        # every existing SiLU chimera graph's digest for nothing.  (The
        # composite MoE node below declares them unconditionally; that shape is
        # sealed by ``NOTES-W12F.md`` and its operator cross-checks both betas
        # against the live module on every crossing.)
        situ_constants = (
            {
                "situ_gate_beta": float(geometry["situ_gate_beta"]),
                "situ_up_beta": float(geometry["situ_up_beta"]),
            }
            if activation == "situ"
            else {}
        )
        if dense:
            require_parameter_set(
                path,
                _CHIMERA_DENSE_FEED_FORWARD_PARAMETERS,
                "dense channel mixer",
                "dense",
            )
            require_linear(
                f"{path}.up",
                (intermediate * 2, hidden),
                f"layer {layer_index} feed-forward up",
            )
            require_linear(
                f"{path}.down",
                (hidden, intermediate),
                f"layer {layer_index} feed-forward down",
            )
            node = builder.add_node(
                f"{prefix}.feed_forward",
                CircuitKind.GATED_MLP,
                path,
                layer_index=layer_index,
                inputs=(normalized_port("residual"),),
                outputs=(delta_port("delta"),),
                role_prefix="feed_forward",
                attributes={
                    **full_width_attributes,
                    **situ_constants,
                    "channel_mixer": "dense",
                },
            )
            builder.edge(
                feed_forward_norm, "normalized", node, "residual", EdgeKind.DATA
            )
            return node, "delta"

        experts = int(geometry["moe_num_experts"])
        shared_experts = int(geometry["moe_shared_experts"])
        latent = int(geometry["moe_latent_size"])
        routed_intermediate = int(geometry["moe_routed_intermediate_size"]) or max(
            1, min(intermediate, latent * 4)
        )
        expected = {
            "router.weight",
            "latent_down.weight",
            "routed_norm.weight",
            "latent_up.weight",
        }
        expected.update(
            f"routed_experts.{index}.{name}.weight"
            for index in range(experts)
            for name in ("up", "down")
        )
        expected.update(
            f"shared_experts.{index}.{name}.weight"
            for index in range(shared_experts)
            for name in ("up", "down")
        )
        require_parameter_set(
            path,
            frozenset(expected),
            "Stable LatentMoE channel mixer",
            "stable_latent_moe",
        )
        require_linear(f"{path}.router", (experts, hidden), "MoE router")
        require_linear(f"{path}.latent_down", (latent, hidden), "MoE latent_down")
        require_rms_norm(f"{path}.routed_norm", latent, "MoE routed_norm")
        require_linear(f"{path}.latent_up", (hidden, latent), "MoE latent_up")
        for expert_index in range(experts):
            expert_path = f"{path}.routed_experts.{expert_index}"
            require_linear(
                f"{expert_path}.up",
                (routed_intermediate * 2, latent),
                f"MoE routed expert {expert_index} up",
            )
            require_linear(
                f"{expert_path}.down",
                (latent, routed_intermediate),
                f"MoE routed expert {expert_index} down",
            )
        for expert_index in range(shared_experts):
            expert_path = f"{path}.shared_experts.{expert_index}"
            require_linear(
                f"{expert_path}.up",
                (intermediate * 2, hidden),
                f"MoE shared expert {expert_index} up",
            )
            require_linear(
                f"{expert_path}.down",
                (hidden, intermediate),
                f"MoE shared expert {expert_index} down",
            )
        # One composite operator, not a router node plus one node per expert
        # plus two residual adds.  ``StableLatentMoEFeedForward.forward``
        # returns ``latent_up(routed_norm(routed)) + sum(shared(x))`` -- one
        # value, written into the residual stream exactly once
        # (``layers.py:428-445``).  The old decomposition encoded two residual
        # writes the model does not perform and could not satisfy the
        # family-neutral one-norm/one-operator/one-write branch contract
        # (``dynamic_fold_planner.py:1026``), which is the same resolution
        # ``NOTES-W12C.md`` section 1 applied to csa/hca.  The sub-branch
        # structure survives as declared attributes, which is what it is:
        # internal structure of one operator.
        node = builder.add_node(
            f"{prefix}.feed_forward",
            CircuitKind.GATED_MLP,
            path,
            layer_index=layer_index,
            inputs=(normalized_port("residual"),),
            outputs=(delta_port("delta"),),
            role_prefix="feed_forward",
            attributes={
                **full_width_attributes,
                "channel_mixer": "stable_latent_moe",
                "composite_operator": True,
                "composite_sub_branches": (
                    "top_k_routed_latent_experts,full_width_shared_experts"
                ),
                "composite_residual_writes": 1,
                "sub_branch_combination": (
                    "sum_of_routed_latent_projection_and_full_width_shared_experts"
                ),
                "expert_count": experts,
                "experts_per_token": int(geometry["moe_experts_per_token"]),
                "shared_expert_count": shared_experts,
                "expert_latent_size": latent,
                "routed_intermediate_size": routed_intermediate,
                "router_score_activation": "sigmoid",
                "router_bias_is_non_gradient_buffer": True,
                "routed_weight_normalization": "selected_score_sum_clamped",
                "router_balancing": "quantile_balanced_between_optimizer_steps",
                "routed_expert_input": "shared_latent_projection_of_the_branch_norm",
                "situ_gate_beta": float(geometry["situ_gate_beta"]),
                "situ_up_beta": float(geometry["situ_up_beta"]),
                "fold_kernel_family_required": (
                    _CHIMERA_STABLE_LATENT_MOE_KERNEL_FAMILY
                ),
            },
        )
        builder.edge(feed_forward_norm, "normalized", node, "residual", EdgeKind.DATA)
        return node, "delta"

    # -- compilation --------------------------------------------------------

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
            raise ChimeraOmniInspectionError(
                "chimera-omni inspection requires PyTorch."
            ) from exc

        if not self.supports(model):
            raise ChimeraOmniInspectionError(
                "chimera-omni inspection requires a ChimeraOmniModel exposing a "
                "chimera config plus embed, blocks, lm_head, native_media_bridge "
                "and native_media_head."
            )
        config = self._config(model)
        manifest = self._manifest(config)
        geometry = self._geometry(manifest, config)
        hidden = geometry["hidden_size"]
        vocab = geometry["vocab_size"]
        feature_dim = geometry["native_feature_dim"]
        position_dim = geometry["native_position_dim"]
        layers = geometry["num_layers"]
        media_dimension = CHIMERA_OMNI_MEDIA_TOKEN_DIMENSION

        try:
            block_modules = tuple(getattr(model, "blocks"))
        except TypeError as exc:
            raise ChimeraOmniInspectionError(
                "chimera-omni blocks container is not iterable."
            ) from exc
        if len(block_modules) != layers:
            raise ChimeraOmniInspectionError(
                f"chimera-omni declares {layers} layers but owns "
                f"{len(block_modules)} blocks."
            )

        def submodule(path: str, label: str) -> Any:
            try:
                return model.get_submodule(path)
            except (AttributeError, KeyError) as exc:
                raise ChimeraOmniInspectionError(
                    f"chimera-omni {label} is missing at {path!r}."
                ) from exc

        def require_linear(path: str, shape: tuple[int, int], label: str) -> None:
            module = submodule(path, label)
            if (
                not isinstance(module, nn.Linear)
                or tuple(int(value) for value in module.weight.shape) != shape
                or module.bias is not None
            ):
                observed: Any = (
                    tuple(int(value) for value in module.weight.shape)
                    if isinstance(module, nn.Linear)
                    else _qualified_class(module)
                )
                raise ChimeraOmniInspectionError(
                    f"chimera-omni {label} must be a bias-free Linear{shape}; "
                    f"observed {observed!r}."
                )

        def require_rms_norm(path: str, width: int, label: str) -> None:
            module = submodule(path, label)
            values = tuple(
                module.named_parameters(recurse=True, remove_duplicate=False)
            )
            if (
                len(values) != 1
                or values[0][0] != "weight"
                or tuple(int(size) for size in values[0][1].shape) != (width,)
            ):
                raise ChimeraOmniInspectionError(
                    f"chimera-omni {label} must own exactly one width-{width} "
                    "RMSNorm scale."
                )

        def require_parameter_set(
            path: str,
            expected: frozenset[str],
            label: str,
            family: str,
        ) -> None:
            module = submodule(path, label)
            observed = frozenset(
                name
                for name, _value in module.named_parameters(
                    recurse=True, remove_duplicate=False
                )
            )
            if observed != expected:
                missing = sorted(expected - observed)
                extra = sorted(observed - expected)
                raise ChimeraOmniInspectionError(
                    f"chimera-omni {family!r} block at {path!r} owns a tensor set "
                    "its declared family does not predict; "
                    f"missing={missing!r}, unexpected={extra!r}."
                )

        # -- interfaces the whole graph depends on --------------------------
        embedding_module = submodule("embed", "token embedding")
        if not isinstance(embedding_module, nn.Embedding) or tuple(
            int(value) for value in embedding_module.weight.shape
        ) != (vocab, hidden):
            raise ChimeraOmniInspectionError(
                "chimera-omni token embedding must be an "
                f"Embedding({vocab}, {hidden})."
            )
        require_rms_norm("norm", hidden, "trunk final norm")
        require_linear("lm_head", (vocab, hidden), "text output head")
        require_linear(
            "native_media_bridge.feature_proj",
            (hidden, feature_dim),
            "native media boundary projection",
        )
        require_linear(
            "native_media_bridge.position_proj",
            (hidden, position_dim),
            "native media position projection",
        )
        require_rms_norm("native_media_bridge.norm", hidden, "native media bridge norm")
        type_embedding = submodule(
            "native_media_bridge.type_embed", "native media type embedding"
        )
        if not isinstance(type_embedding, nn.Embedding) or tuple(
            int(value) for value in type_embedding.weight.shape
        ) != (geometry["native_type_count"], hidden):
            raise ChimeraOmniInspectionError(
                "chimera-omni native media type embedding must be an "
                f"Embedding({geometry['native_type_count']}, {hidden})."
            )
        require_rms_norm("native_media_head.norm", hidden, "native media emission norm")
        require_linear(
            "native_media_head.out",
            (feature_dim, hidden),
            "native media emission head",
        )
        try:
            mtp_modules = tuple(getattr(model, "mtp_heads", ()))
        except TypeError as exc:
            raise ChimeraOmniInspectionError(
                "chimera-omni mtp_heads container is not iterable."
            ) from exc
        if len(mtp_modules) != geometry["mtp_heads"]:
            raise ChimeraOmniInspectionError(
                f"chimera-omni declares {geometry['mtp_heads']} MTP heads but "
                f"owns {len(mtp_modules)}."
            )
        for index in range(len(mtp_modules)):
            require_linear(f"mtp_heads.{index}", (vocab, hidden), f"MTP head {index}")
        auxiliary = (
            ("flow", "flow_head", 4),
            ("grounding", "grounding_head", 4),
            ("sync", "sync_head", 1),
        )
        declared_aux = set(geometry["aux_heads"])
        unknown_aux = declared_aux - {name for name, _path, _width in auxiliary}
        if unknown_aux:
            raise ChimeraOmniInspectionError(
                "chimera-omni declares auxiliary heads with no inspected "
                "decomposition: " + ", ".join(sorted(unknown_aux)) + "."
            )
        active_aux: list[tuple[str, str, int]] = []
        for aux_name, attribute, width in auxiliary:
            module = getattr(model, attribute, None)
            if (aux_name in declared_aux) != (module is not None):
                raise ChimeraOmniInspectionError(
                    f"chimera-omni auxiliary head {aux_name!r} is declared "
                    f"{aux_name in declared_aux} but its module is "
                    f"{'present' if module is not None else 'absent'}."
                )
            if module is not None:
                require_linear(attribute, (width, hidden), f"{aux_name} head")
                active_aux.append((aux_name, attribute, width))

        residual_mode = geometry["residual_mode"]
        attnres_block_size = geometry["attnres_block_size"]
        output_depth_module = getattr(model, "output_attention_residual", None)
        if residual_mode == "block_attnres":
            if attnres_block_size <= 0 or output_depth_module is None:
                raise ChimeraOmniInspectionError(
                    "chimera-omni block_attnres requires a positive "
                    "attnres_block_size and a model-level depth residual."
                )
            checkpoint_layers = tuple(
                index for index in range(layers) if index % attnres_block_size == 0
            )
        else:
            if output_depth_module is not None:
                raise ChimeraOmniInspectionError(
                    "chimera-omni residual_mode 'none' must not own a model-level "
                    "depth residual selector."
                )
            checkpoint_layers = ()

        view = _ChimeraOmniLayoutView(model, manifest)
        groups = _storage_groups(view)
        builder = _GraphBuilder(
            model=view,
            model_name=model_name,
            family=CHIMERA_OMNI_FAMILY,
            tokenizer_fingerprint=tokenizer_fingerprint,
            groups=groups,
            nodes=[],
            edges=[],
            interfaces=[],
            captured=set(),
        )

        def residual_port(name: str) -> CircuitPort:
            return _port(name, PortKind.RESIDUAL, "batch", "sequence", hidden)

        def normalized_port(name: str) -> CircuitPort:
            return _port(name, PortKind.NORMALIZED_RESIDUAL, "batch", "sequence", hidden)

        def delta_port(name: str) -> CircuitPort:
            return _port(name, PortKind.BRANCH_DELTA, "batch", "sequence", hidden)

        def media_port(
            name: str,
            width: int,
            modality: str | None = None,
        ) -> CircuitPort:
            return _port(
                name,
                PortKind.MODALITY,
                "batch",
                media_dimension,
                width,
                modality=modality,
            )

        # -- token entry ----------------------------------------------------
        token_node = builder.add_node(
            "input.token_embedding",
            CircuitKind.TOKEN_EMBEDDING,
            "embed",
            layer_index=None,
            inputs=(_port("token_ids", PortKind.TOKEN_IDS, "batch", "sequence"),),
            outputs=(
                _port(
                    "residual",
                    PortKind.RESIDUAL,
                    "batch",
                    "sequence",
                    hidden,
                    modality=CHIMERA_OMNI_TEXT_MODALITY,
                ),
            ),
            role_prefix="token_embedding",
            attributes={
                "vocabulary_size": vocab,
                "hidden_size": hidden,
                "typed_token_ledger": True,
                "ledger_ranges": "text,control,media,grounding,sync,tool",
            },
        )
        builder.interfaces.append(
            ModalityInterface(
                interface_id="text.tokens.input",
                modality=CHIMERA_OMNI_TEXT_MODALITY,
                direction="input",
                node_id=token_node.node_id,
                port_name="token_ids",
                representation="token_ids",
            )
        )

        # -- encoder-free native media boundary projection ------------------
        lane_names = ",".join(
            lane for lane, _type_id, _canonical in CHIMERA_OMNI_MEDIA_LANES
        )
        boundary_node = builder.add_node(
            "input.native_media.boundary_projection",
            CircuitKind.MODALITY_INPUT,
            "native_media_bridge.feature_proj",
            layer_index=None,
            inputs=tuple(
                media_port(f"{lane}_features", feature_dim, canonical)
                for lane, _type_id, canonical in CHIMERA_OMNI_MEDIA_LANES
            ),
            outputs=(media_port("projected", hidden),),
            role_prefix="native_media_boundary_projection",
            attributes={
                "native_feature_dim": feature_dim,
                "hidden_size": hidden,
                "shared_projection_for_every_modality": True,
                "modality_tower_count": 0,
                "external_frozen_encoder": False,
                "learned_tensor_count": 1,
                "edge_patchify_or_segmentize": True,
                "media_lanes": lane_names,
                "lane_canonicalization": CHIMERA_OMNI_LANE_CANONICALIZATION,
            },
        )
        for lane, _type_id, canonical in CHIMERA_OMNI_MEDIA_LANES:
            builder.interfaces.append(
                ModalityInterface(
                    interface_id=f"native_media.{lane}.input",
                    modality=canonical,
                    direction="input",
                    node_id=boundary_node.node_id,
                    port_name=f"{lane}_features",
                    representation=f"chimera_native_continuous_segments_{lane}",
                )
            )
        type_node = builder.add_node(
            "input.native_media.type_embedding",
            CircuitKind.MODALITY_INPUT,
            "native_media_bridge.type_embed",
            layer_index=None,
            inputs=(
                media_port("projected", hidden),
                _port("type_ids", PortKind.GENERIC, "batch", media_dimension),
            ),
            outputs=(media_port("typed", hidden),),
            role_prefix="native_media_type_embedding",
            attributes={
                "native_type_count": geometry["native_type_count"],
                "ledger_range": "media",
                "type_ids": ",".join(
                    f"{lane}={type_id}"
                    for lane, type_id, _canonical in CHIMERA_OMNI_MEDIA_LANES
                ),
                "clamped_to_table": True,
            },
        )
        position_node = builder.add_node(
            "input.native_media.position_projection",
            CircuitKind.POSITION_ENCODING,
            "native_media_bridge.position_proj",
            layer_index=None,
            inputs=(
                media_port("typed", hidden),
                _port(
                    "positions",
                    PortKind.GENERIC,
                    "batch",
                    media_dimension,
                    position_dim,
                ),
            ),
            outputs=(media_port("positioned", hidden),),
            role_prefix="native_media_position_projection",
            attributes={
                "kind": "continuous_native_position_projection",
                "native_position_dim": position_dim,
                "learned_absolute_table": False,
            },
        )
        bridge_norm_node = builder.add_node(
            "input.native_media.bridge_norm",
            CircuitKind.NORM,
            "native_media_bridge.norm",
            layer_index=None,
            inputs=(media_port("positioned", hidden),),
            outputs=(media_port("media_residual", hidden),),
            role_prefix="native_media_bridge_norm",
            attributes={"norm_kind": "rms_norm", "weight_kind": "direct_scale"},
        )
        fusion_node = builder.add_node(
            "input.native_media_text_fusion",
            CircuitKind.INTERFACE_PROJECTION,
            "native_media_bridge",
            layer_index=None,
            inputs=(
                _port(
                    "text_residual",
                    PortKind.RESIDUAL,
                    "batch",
                    "sequence",
                    hidden,
                    modality=CHIMERA_OMNI_TEXT_MODALITY,
                ),
                media_port("media_residual", hidden),
            ),
            outputs=(residual_port("residual"),),
            attributes={
                "operation": "scatter_add_native_media_rows_into_trunk_residual",
                "alignment_rule": (
                    "media token ids occupy trunk positions; continuous media "
                    "features add at those positions"
                ),
                "learned_parameters": 0,
                "optional_at_runtime": True,
            },
        )
        builder.edge(
            boundary_node, "projected", type_node, "projected", EdgeKind.MODALITY
        )
        builder.edge(type_node, "typed", position_node, "typed", EdgeKind.MODALITY)
        builder.edge(
            position_node,
            "positioned",
            bridge_norm_node,
            "positioned",
            EdgeKind.MODALITY,
        )
        builder.edge(
            bridge_norm_node,
            "media_residual",
            fusion_node,
            "media_residual",
            EdgeKind.MODALITY,
        )
        builder.edge(
            token_node, "residual", fusion_node, "text_residual", EdgeKind.TOKEN
        )

        # -- trunk ----------------------------------------------------------
        checkpoint_producers: dict[int, tuple[CircuitNode, str]] = {}
        if 0 in checkpoint_layers:
            checkpoint_producers[0] = (fusion_node, "residual")
        current = fusion_node
        current_port = "residual"
        block_kinds: list[str] = []
        mixer_kernel_families: set[str] = set()

        for index, block in enumerate(block_modules):
            declared = geometry["block_pattern"][index % len(geometry["block_pattern"])]
            observed_kind = getattr(block, "kind", None)
            if (
                not isinstance(observed_kind, str)
                or observed_kind not in CHIMERA_OMNI_BLOCK_KINDS
            ):
                raise ChimeraOmniInspectionError(
                    f"chimera-omni block {index} runs sequence-mixing family "
                    f"{observed_kind!r}, which has no inspected decomposition; "
                    "the supported families are "
                    + ", ".join(CHIMERA_OMNI_BLOCK_KINDS)
                    + "."
                )
            if observed_kind != declared and not (
                geometry["force_final_gated_mla"]
                and index == layers - 1
                and observed_kind == "gated_mla"
            ):
                raise ChimeraOmniInspectionError(
                    f"chimera-omni block {index} runs family {observed_kind!r} "
                    f"but the block pattern declares {declared!r}."
                )
            block_kinds.append(observed_kind)
            mixer_kernel_families.add(_CHIMERA_MIXER_KERNEL_FAMILIES[observed_kind])
            prefix = f"layer.{index:03d}"
            block_path = f"blocks.{index}"
            require_rms_norm(f"{block_path}.norm1", hidden, f"layer {index} norm1")
            require_rms_norm(f"{block_path}.norm2", hidden, f"layer {index} norm2")

            operator_input = current
            operator_input_port = current_port
            if residual_mode == "block_attnres":
                slots = tuple(
                    position
                    for position, checkpoint in enumerate(checkpoint_layers)
                    if checkpoint < index
                )
                operator_input = self._depth_select_node(
                    builder,
                    node_id=f"{prefix}.attention_depth_select",
                    path=f"{block_path}.attention_residual",
                    role_prefix="depth_attention_residual",
                    layer_index=index,
                    hidden=hidden,
                    source_count=len(slots),
                    residual_port=residual_port,
                    attributes={"branch": "operator"},
                )
                builder.edge(
                    current,
                    current_port,
                    operator_input,
                    "residual",
                    EdgeKind.RESIDUAL,
                )
                for slot, position in enumerate(slots):
                    producer, producer_port = checkpoint_producers[
                        checkpoint_layers[position]
                    ]
                    builder.edge(
                        producer,
                        producer_port,
                        operator_input,
                        f"depth_source_{slot:03d}",
                        EdgeKind.RESIDUAL,
                    )
                operator_input_port = "selected"

            operator_norm = builder.add_node(
                f"{prefix}.operator_norm",
                CircuitKind.NORM,
                f"{block_path}.norm1",
                layer_index=index,
                inputs=(residual_port("residual"),),
                outputs=(normalized_port("normalized"),),
                role_prefix="operator_norm",
                attributes={"norm_kind": "rms_norm", "weight_kind": "direct_scale"},
            )
            builder.edge(
                operator_input,
                operator_input_port,
                operator_norm,
                "residual",
                EdgeKind.DATA,
            )
            delta_source, delta_source_port = self._mixer_nodes(
                builder,
                model=model,
                prefix=prefix,
                block_path=block_path,
                block_kind=observed_kind,
                layer_index=index,
                geometry=geometry,
                normalized_port=normalized_port,
                delta_port=delta_port,
                require_linear=require_linear,
                require_rms_norm=require_rms_norm,
                require_parameter_set=require_parameter_set,
                operator_norm=operator_norm,
            )
            # ``block_attnres`` is a *block* scheme: at every checkpoint depth
            # the running prefix sum is handed to the depth-source list and the
            # accumulator restarts from this branch's own output.  There is no
            # base term to add at those depths, so the write node genuinely has
            # one input, and saying otherwise would put an edge in the graph
            # that the model never executes.
            accumulator_reset = index in checkpoint_layers
            operator_write = builder.add_node(
                f"{prefix}.operator_residual",
                CircuitKind.RESIDUAL_ADD,
                block_path,
                layer_index=index,
                inputs=(
                    (delta_port("delta"),)
                    if accumulator_reset
                    else (residual_port("base"), delta_port("delta"))
                ),
                outputs=(residual_port("residual"),),
                attributes={
                    "ordered_branch": "operator",
                    "residual_accumulator_reset": accumulator_reset,
                    "depth_checkpoint_taken_before_this_write": accumulator_reset,
                },
            )
            if not accumulator_reset:
                builder.edge(
                    current, current_port, operator_write, "base", EdgeKind.RESIDUAL
                )
            builder.edge(
                delta_source,
                delta_source_port,
                operator_write,
                "delta",
                EdgeKind.RESIDUAL,
            )
            current = operator_write
            current_port = "residual"

            feed_forward_input = current
            feed_forward_input_port = current_port
            if residual_mode == "block_attnres":
                slots = tuple(
                    position
                    for position, checkpoint in enumerate(checkpoint_layers)
                    if checkpoint <= index
                )
                feed_forward_input = self._depth_select_node(
                    builder,
                    node_id=f"{prefix}.feed_forward_depth_select",
                    path=f"{block_path}.feed_forward_residual",
                    role_prefix="depth_attention_residual",
                    layer_index=index,
                    hidden=hidden,
                    source_count=len(slots),
                    residual_port=residual_port,
                    attributes={"branch": "feed_forward"},
                )
                builder.edge(
                    current,
                    current_port,
                    feed_forward_input,
                    "residual",
                    EdgeKind.RESIDUAL,
                )
                for slot, position in enumerate(slots):
                    producer, producer_port = checkpoint_producers[
                        checkpoint_layers[position]
                    ]
                    builder.edge(
                        producer,
                        producer_port,
                        feed_forward_input,
                        f"depth_source_{slot:03d}",
                        EdgeKind.RESIDUAL,
                    )
                feed_forward_input_port = "selected"

            feed_forward_norm = builder.add_node(
                f"{prefix}.feed_forward_norm",
                CircuitKind.NORM,
                f"{block_path}.norm2",
                layer_index=index,
                inputs=(residual_port("residual"),),
                outputs=(normalized_port("normalized"),),
                role_prefix="feed_forward_norm",
                attributes={"norm_kind": "rms_norm", "weight_kind": "direct_scale"},
            )
            builder.edge(
                feed_forward_input,
                feed_forward_input_port,
                feed_forward_norm,
                "residual",
                EdgeKind.DATA,
            )
            channel_source, channel_source_port = self._channel_mixer_nodes(
                builder,
                prefix=prefix,
                block_path=block_path,
                layer_index=index,
                geometry=geometry,
                normalized_port=normalized_port,
                delta_port=delta_port,
                require_linear=require_linear,
                require_rms_norm=require_rms_norm,
                require_parameter_set=require_parameter_set,
                feed_forward_norm=feed_forward_norm,
            )
            feed_forward_write = builder.add_node(
                f"{prefix}.feed_forward_residual",
                CircuitKind.RESIDUAL_ADD,
                block_path,
                layer_index=index,
                inputs=(residual_port("base"), delta_port("delta")),
                outputs=(residual_port("residual"),),
                attributes={"ordered_branch": "feed_forward"},
            )
            builder.edge(
                current, current_port, feed_forward_write, "base", EdgeKind.RESIDUAL
            )
            builder.edge(
                channel_source,
                channel_source_port,
                feed_forward_write,
                "delta",
                EdgeKind.RESIDUAL,
            )
            current = feed_forward_write
            current_port = "residual"
            if index + 1 in checkpoint_layers:
                checkpoint_producers[index + 1] = (current, current_port)

        # -- output side ----------------------------------------------------
        if residual_mode == "block_attnres":
            output_select = self._depth_select_node(
                builder,
                node_id="output.depth_select",
                path="output_attention_residual",
                role_prefix="output_depth_attention_residual",
                layer_index=None,
                hidden=hidden,
                source_count=len(checkpoint_layers),
                residual_port=residual_port,
                attributes={"branch": "trunk_output"},
            )
            builder.edge(
                current, current_port, output_select, "residual", EdgeKind.RESIDUAL
            )
            for slot, checkpoint in enumerate(checkpoint_layers):
                producer, producer_port = checkpoint_producers[checkpoint]
                builder.edge(
                    producer,
                    producer_port,
                    output_select,
                    f"depth_source_{slot:03d}",
                    EdgeKind.RESIDUAL,
                )
            current = output_select
            current_port = "selected"

        final_norm = builder.add_node(
            "output.final_norm",
            CircuitKind.NORM,
            "norm",
            layer_index=None,
            inputs=(residual_port("residual"),),
            outputs=(normalized_port("normalized"),),
            role_prefix="final_norm",
            attributes={"norm_kind": "rms_norm", "weight_kind": "direct_scale"},
        )
        builder.edge(current, current_port, final_norm, "residual", EdgeKind.DATA)

        head = builder.add_node(
            "output.lm_head",
            CircuitKind.OUTPUT_HEAD,
            "lm_head",
            layer_index=None,
            inputs=(normalized_port("residual"),),
            outputs=(
                _port(
                    "logits",
                    PortKind.LOGITS,
                    "batch",
                    "sequence",
                    vocab,
                    modality=CHIMERA_OMNI_TEXT_MODALITY,
                ),
            ),
            role_prefix="output_head",
            attributes={
                "vocabulary_size": vocab,
                "tied_to_embedding": False,
                "head_role": "next_token",
            },
        )
        builder.edge(final_norm, "normalized", head, "residual", EdgeKind.DATA)
        builder.interfaces.append(
            ModalityInterface(
                interface_id="text.logits.output",
                modality=CHIMERA_OMNI_TEXT_MODALITY,
                direction="output",
                node_id=head.node_id,
                port_name="logits",
                representation="token_logits",
            )
        )
        for index in range(len(mtp_modules)):
            mtp_node = builder.add_node(
                f"output.mtp_head.{index:03d}",
                CircuitKind.OUTPUT_HEAD,
                f"mtp_heads.{index}",
                layer_index=None,
                inputs=(normalized_port("residual"),),
                outputs=(
                    _port(
                        "logits",
                        PortKind.LOGITS,
                        "batch",
                        "sequence",
                        vocab,
                        modality=CHIMERA_OMNI_TEXT_MODALITY,
                    ),
                ),
                role_prefix="multi_token_prediction_head",
                attributes={
                    "vocabulary_size": vocab,
                    "head_role": "multi_token_prediction",
                    "predicted_offset": index + 2,
                },
            )
            builder.edge(final_norm, "normalized", mtp_node, "residual", EdgeKind.DATA)
        for aux_name, attribute, width in active_aux:
            aux_node = builder.add_node(
                f"output.aux.{aux_name}",
                CircuitKind.OUTPUT_HEAD,
                attribute,
                layer_index=None,
                inputs=(normalized_port("residual"),),
                outputs=(_port("values", PortKind.GENERIC, "batch", "sequence", width),),
                role_prefix="auxiliary_head",
                attributes={"head_role": aux_name, "output_width": width},
            )
            builder.edge(final_norm, "normalized", aux_node, "residual", EdgeKind.DATA)

        # -- the shared native media emission head --------------------------
        # The repository's second real MODALITY_OUTPUT emitter.  It reads the
        # same final-normed rows the text head reads, so the two heads are
        # siblings: emitted media is an additional output of the trunk, not a
        # post-processor of text.
        gather_node = builder.add_node(
            "output.native_media.gather",
            CircuitKind.INTERFACE_PROJECTION,
            "native_media_head",
            layer_index=None,
            inputs=(normalized_port("residual"),),
            outputs=(media_port("media_rows", hidden),),
            attributes={
                "operation": "gather_native_media_rows_from_trunk_hidden",
                "index_source": "native_media_indices",
                "learned_parameters": 0,
            },
        )
        builder.edge(final_norm, "normalized", gather_node, "residual", EdgeKind.DATA)
        emission_norm = builder.add_node(
            "output.native_media.norm",
            CircuitKind.NORM,
            "native_media_head.norm",
            layer_index=None,
            inputs=(media_port("media_rows", hidden),),
            outputs=(media_port("normalized", hidden),),
            role_prefix="native_media_emission_norm",
            attributes={"norm_kind": "rms_norm", "weight_kind": "direct_scale"},
        )
        builder.edge(
            gather_node, "media_rows", emission_norm, "media_rows", EdgeKind.MODALITY
        )
        emission_head = builder.add_node(
            "output.native_media.emission_head",
            CircuitKind.MODALITY_OUTPUT,
            "native_media_head.out",
            layer_index=None,
            inputs=(media_port("normalized", hidden),),
            outputs=tuple(
                media_port(f"{lane}_features", feature_dim, canonical)
                for lane, _type_id, canonical in CHIMERA_OMNI_MEDIA_LANES
            ),
            role_prefix="native_media_emission_head",
            attributes={
                "native_feature_dim": feature_dim,
                "shared_emission_head_for_every_modality": True,
                "modality_tower_count": 0,
                "emitted_encoding": "continuous_shared_reconstruction",
                "reduction": "native_media_rows",
                "learned_tensor_count": 1,
                "media_lanes": lane_names,
                "lane_canonicalization": CHIMERA_OMNI_LANE_CANONICALIZATION,
            },
        )
        builder.edge(
            emission_norm, "normalized", emission_head, "normalized", EdgeKind.MODALITY
        )
        for lane, _type_id, canonical in CHIMERA_OMNI_MEDIA_LANES:
            builder.interfaces.append(
                ModalityInterface(
                    interface_id=f"native_media.{lane}.output",
                    modality=canonical,
                    direction="output",
                    node_id=emission_head.node_id,
                    port_name=f"{lane}_features",
                    representation=f"chimera_native_continuous_reconstruction_{lane}",
                )
            )

        present_kinds = set(block_kinds)
        graph = builder.finish(
            metadata={
                "inspector": self.name,
                "architecture": str(manifest["architecture"]),
                "position_policy": CHIMERA_OMNI_POSITION_POLICY,
                "hidden_size": hidden,
                "vocab_size": vocab,
                "layer_count": layers,
                "intermediate_size": geometry["intermediate_size"],
                "num_heads": geometry["num_heads"],
                "num_kv_heads": geometry["num_kv_heads"],
                "attention_head_dim": geometry["head_dim"],
                "local_window": geometry["local_window"],
                "csa_compress_stride": geometry["csa_compress_stride"],
                "hca_compress_stride": geometry["hca_compress_stride"],
                "block_pattern": ",".join(geometry["block_pattern"]),
                "block_kinds": ",".join(block_kinds),
                "residual_mode": residual_mode,
                "attnres_block_size": attnres_block_size,
                "depth_axis_block_attnres": residual_mode == "block_attnres",
                "depth_checkpoint_layers": ",".join(
                    str(value) for value in checkpoint_layers
                ),
                "feed_forward_kind": geometry["feed_forward_kind"],
                "activation_kind": geometry["activation_kind"],
                "native_feature_dim": feature_dim,
                "native_position_dim": position_dim,
                "native_type_count": geometry["native_type_count"],
                "mtp_head_count": len(mtp_modules),
                "aux_heads": ",".join(geometry["aux_heads"]),
                "encoder_free_native_media": True,
                "external_frozen_media_encoder": False,
                "modality_tower_count": 0,
                "shared_native_media_projection": True,
                "shared_native_media_emission_head": True,
                "emits_native_media": True,
                "native_media_scatter_fusion": True,
                "chimera_media_lane_canonicalization": (
                    CHIMERA_OMNI_LANE_CANONICALIZATION
                ),
                "mixer_kernel_families": ",".join(sorted(mixer_kernel_families)),
                "structurally_coverable_block_kinds": ",".join(
                    sorted(present_kinds & _CHIMERA_STRUCTURALLY_COVERABLE_MIXERS)
                ),
                "kernel_gap_block_kinds": ",".join(
                    sorted(present_kinds - _CHIMERA_STRUCTURALLY_COVERABLE_MIXERS)
                ),
                "tie_word_embeddings": False,
                "wrapped_model_class": _qualified_class(model),
            }
        )
        return _chimera_graph_with_model_class(graph, _qualified_class(model))


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
            NativeVideoCircuitInspector(),
            NativePoints3dCircuitInspector(),
            SmolVlmCircuitInspector(),
            ChimeraOmniCircuitInspector(),
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


# ---------------------------------------------------------------------------
# Bare tied decoder bodies (ACE-Step 5Hz audio-token LM lane).
#
# Some checkpoints publish the decoder *body* only: ``architectures`` reads
# ``["Qwen3Model"]``, no ``lm_head`` module exists, and the checkpoint keys
# carry no ``model.`` prefix.  A bare body is not a causal LM, so the decoder
# layout contract refuses it -- correctly, and that refusal is left untouched.
#
# When such a body declares tied word embeddings, the runtime reads logits
# straight off the input embedding, so the equivalent causal LM is already
# fully determined by the body: ``logits = h @ E^T``.  This adapter materialises
# exactly that wrapper so the *existing* inspector and evidence paths can run
# against the real weights.  It never invents head parameters: an untied body
# has no recoverable head, and every unmet precondition fails closed.
# ---------------------------------------------------------------------------


_BARE_TIED_CAUSAL_BODIES: Mapping[str, tuple[str, str]] = {
    "qwen3": ("Qwen3Model", "Qwen3ForCausalLM"),
}


def wrap_bare_tied_causal_body(model: Any) -> Any:
    """Wrap a bare tied decoder body in its equivalent causal-LM shell.

    The returned wrapper shares -- never copies -- the body's parameters, and
    its ``lm_head.weight`` is the very same tensor object as the body's
    ``embed_tokens.weight``, which is what tied word embeddings mean.
    """

    import torch

    config = getattr(model, "config", None)
    if config is None:
        raise DynamicModelInspectionError(
            "A bare tied causal body must carry a Hugging Face configuration."
        )
    model_type = getattr(config, "model_type", None)
    if model_type not in _BARE_TIED_CAUSAL_BODIES:
        supported = ", ".join(sorted(_BARE_TIED_CAUSAL_BODIES))
        raise DynamicModelInspectionError(
            f"Bare tied causal body wrapping supports model_type {supported}; "
            f"got {model_type!r}."
        )
    body_name, wrapper_name = _BARE_TIED_CAUSAL_BODIES[model_type]
    observed = type(model).__name__
    if observed != body_name:
        raise DynamicModelInspectionError(
            f"Expected a bare {body_name} body to wrap, got {observed}."
        )
    if getattr(config, "tie_word_embeddings", None) is not True:
        raise DynamicModelInspectionError(
            "Only a tied bare body determines its own head; "
            f"{observed} declares tie_word_embeddings="
            f"{getattr(config, 'tie_word_embeddings', None)!r}."
        )
    if getattr(model, "lm_head", None) is not None:
        raise DynamicModelInspectionError(
            f"{observed} already exposes an lm_head and is not a bare body."
        )
    output_embeddings = getattr(model, "get_output_embeddings", None)
    if callable(output_embeddings) and output_embeddings() is not None:
        raise DynamicModelInspectionError(
            f"{observed} already exposes output embeddings and is not a bare body."
        )
    embedding = getattr(model, "embed_tokens", None)
    weight = getattr(embedding, "weight", None)
    if weight is None:
        raise DynamicModelInspectionError(
            f"{observed} exposes no embed_tokens.weight to tie a head onto."
        )
    expected_shape = (
        int(getattr(config, "vocab_size", -1)),
        int(getattr(config, "hidden_size", -1)),
    )
    if tuple(int(value) for value in weight.shape) != expected_shape:
        raise DynamicModelInspectionError(
            "embed_tokens.weight must be (vocab_size, hidden_size)="
            f"{expected_shape}, got {tuple(int(v) for v in weight.shape)}."
        )

    import transformers

    shell = getattr(transformers, wrapper_name)
    # Build the shell on the meta device so wrapping never allocates a second
    # copy of a multi-gigabyte parameter set; every real tensor is then shared
    # in from the body.
    with torch.device("meta"):
        wrapper = shell(config)
    wrapper.model = model
    wrapper.lm_head.weight = weight
    wrapper.eval()

    if wrapper.lm_head.weight is not model.embed_tokens.weight:
        raise DynamicModelInspectionError(
            "The wrapped head is not the body's embedding tensor."
        )
    unplaced = sorted(
        name for name, value in wrapper.named_parameters() if value.is_meta
    )
    if unplaced:
        raise DynamicModelInspectionError(
            "Wrapping left unmaterialised parameters: " + ", ".join(unplaced)
        )
    return wrapper


__all__ = [
    "CHIMERA_OMNI_ARCHITECTURES",
    "CHIMERA_OMNI_BLOCK_KINDS",
    "CHIMERA_OMNI_CONFIG_SCHEMA",
    "CHIMERA_OMNI_FAMILY",
    "CHIMERA_OMNI_FEED_FORWARD_KINDS",
    "CHIMERA_OMNI_LANE_CANONICALIZATION",
    "CHIMERA_OMNI_MEDIA_LANES",
    "CHIMERA_OMNI_MEDIA_TOKEN_DIMENSION",
    "CHIMERA_OMNI_POSITION_POLICY",
    "CHIMERA_OMNI_RESIDUAL_MODES",
    "NATIVE_AUDIO_FAMILY",
    "NATIVE_AUDIO_LANE_CONFIG_KEY",
    "NATIVE_AUDIO_LANE_MARKER",
    "NATIVE_AUDIO_MODEL_TYPE",
    "NATIVE_AUDIO_REPRESENTATION",
    "NATIVE_POINTS3D_EMITTED_REPRESENTATION",
    "NATIVE_POINTS3D_FAMILY",
    "NATIVE_POINTS3D_LANE_CONFIG_KEY",
    "NATIVE_POINTS3D_LANE_MARKER",
    "NATIVE_POINTS3D_MODEL_TYPE",
    "NATIVE_POINTS3D_POOLINGS",
    "NATIVE_POINTS3D_REPRESENTATION",
    "NATIVE_VIDEO_FAMILY",
    "NATIVE_VIDEO_LANE_CONFIG_KEY",
    "NATIVE_VIDEO_LANE_MARKER",
    "NATIVE_VIDEO_MODEL_TYPE",
    "NATIVE_VIDEO_REPRESENTATION",
    "SMOLVLM_FAMILY",
    "SMOLVLM_IMAGE_REPRESENTATION",
    "SMOLVLM_MODEL_CLASS",
    "SMOLVLM_MODEL_TYPE",
    "SMOLVLM_PATCH_MASK_REPRESENTATION",
    "SMOLVLM_VIDEO_NOTE",
    "SMOLVLM_VISION_MODEL_TYPE",
    "ChimeraOmniCircuitInspector",
    "ChimeraOmniInspectionError",
    "DynamicModelInspectionError",
    "HFCausalDecoderInspector",
    "InspectorRegistry",
    "Lfm2CircuitInspector",
    "ModelCircuitInspector",
    "NativeAudioCircuitInspector",
    "NativePoints3dCircuitInspector",
    "NativeVideoCircuitInspector",
    "NeoNativeVisionCircuitInspector",
    "Qwen3CodebookCircuitInspector",
    "Qwen3CompensatedCircuitInspector",
    "SmolVlmCircuitInspector",
    "default_inspector_registry",
    "inspect_model_circuits",
    "wrap_bare_tied_causal_body",
]
