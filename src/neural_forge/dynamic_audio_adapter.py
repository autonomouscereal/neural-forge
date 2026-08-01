"""Local-only exact target synthesis for native audio-text models.

This is the audio mirror of :mod:`neural_forge.dynamic_neo_adapter`.  It clones
one loaded parent's own class and configuration, reduces exactly one field --
``decoder_config.intermediate_size`` -- and proves through independent
re-inspection that nothing else moved.  The learned PCM16 frame encoder is
copied shape-for-shape: an audio front end is never rebuilt, rescaled, or
replaced by an external encoder.
"""

from __future__ import annotations

import copy
import threading
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from .dynamic_circuit_ir import CircuitKind, ModelCircuitGraph, canonical_sha256
from .dynamic_model_inspector import (
    NATIVE_AUDIO_FAMILY,
    NATIVE_AUDIO_LANE_CONFIG_KEY,
    NATIVE_AUDIO_LANE_MARKER,
    NATIVE_AUDIO_MODEL_TYPE,
)
from .dynamic_target_synthesis import (
    AdapterMaterialization,
    DynamicTargetRequest,
)


ADAPTER_SCHEMA = "neural-forge-dynamic-native-audio-adapter-v1"
TARGET_MODALITIES = ("audio", "text")
_DEFAULT_DTYPE_LOCK = threading.RLock()


class DynamicAudioAdapterError(RuntimeError):
    """A native audio target could not be proven from a loaded local parent."""


@contextmanager
def _temporary_default_dtype(torch: Any, dtype: Any):
    with _DEFAULT_DTYPE_LOCK:
        previous = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            yield
        finally:
            torch.set_default_dtype(previous)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DynamicAudioAdapterError(f"{name} must be a positive integer.")
    return value


def _config_payload(config: Any) -> dict[str, Any]:
    to_dict = getattr(config, "to_dict", None)
    if not callable(to_dict):
        raise DynamicAudioAdapterError(
            "Native audio configuration must expose to_dict()."
        )
    payload = to_dict()
    if not isinstance(payload, Mapping):
        raise DynamicAudioAdapterError(
            "Native audio configuration did not serialize a mapping."
        )

    def normalize(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): normalize(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [normalize(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    return normalize(payload)


def _leaf_differences(
    source: Any,
    target: Any,
    prefix: tuple[str, ...] = (),
) -> tuple[tuple[str, ...], ...]:
    if isinstance(source, Mapping) and isinstance(target, Mapping):
        differences = []
        for key in sorted(set(source) | set(target), key=str):
            if key not in source or key not in target:
                differences.append((*prefix, str(key)))
                continue
            differences.extend(
                _leaf_differences(
                    source[key],
                    target[key],
                    (*prefix, str(key)),
                )
            )
        return tuple(differences)
    if source != target:
        return (prefix,)
    return ()


def _graph_topology_payload(graph: ModelCircuitGraph) -> dict[str, Any]:
    """Describe executable topology while excluding weight sizes and identities."""

    return {
        "family": graph.family,
        "model_class": graph.model_class,
        "nodes": [
            {
                "node_id": node.node_id,
                "kind": node.kind,
                "order": node.order,
                "path": node.path,
                "layer_index": node.layer_index,
                "inputs": [port.as_dict() for port in node.inputs],
                "outputs": [port.as_dict() for port in node.outputs],
                "parameters": [
                    {
                        "path": parameter.path,
                        "role": parameter.role,
                        "dtype": parameter.dtype,
                        "trainable": parameter.trainable,
                    }
                    for parameter in node.parameters
                ],
                "attributes": {
                    key: value
                    for key, value in node.attributes
                    if key != "intermediate_size"
                },
            }
            for node in graph.nodes
        ],
        "edges": [edge.as_dict() for edge in graph.edges],
        "interfaces": [interface.as_dict() for interface in graph.interfaces],
        "layer_indices": list(graph.layer_indices),
        "tokenizer_fingerprint": graph.tokenizer_fingerprint,
    }


def _parameter_shapes(graph: ModelCircuitGraph) -> dict[str, tuple[int, ...]]:
    return {
        parameter.path: parameter.shape
        for node in graph.nodes
        for parameter in node.parameters
    }


def _validate_only_decoder_mlp_width_changed(
    parent: ModelCircuitGraph,
    child: ModelCircuitGraph,
    *,
    source_intermediate: int,
    target_intermediate: int,
    hidden_size: int,
) -> int:
    if _graph_topology_payload(parent) != _graph_topology_payload(child):
        raise DynamicAudioAdapterError(
            "Materialized native audio child changed executable topology "
            "outside the requested decoder MLP width."
        )
    if parent.layer_indices != child.layer_indices or not parent.layer_indices:
        raise DynamicAudioAdapterError(
            "Materialized native audio child changed decoder depth."
        )

    parent_shapes = _parameter_shapes(parent)
    child_shapes = _parameter_shapes(child)
    if set(parent_shapes) != set(child_shapes):
        raise DynamicAudioAdapterError(
            "Materialized native audio child changed named parameter ownership."
        )
    allowed_changed_paths: set[str] = set()
    for layer_index in parent.layer_indices:
        parent_nodes = parent.nodes_of_kind(
            CircuitKind.GATED_MLP,
            layer_index=layer_index,
        )
        child_nodes = child.nodes_of_kind(
            CircuitKind.GATED_MLP,
            layer_index=layer_index,
        )
        if len(parent_nodes) != 1 or len(child_nodes) != 1:
            raise DynamicAudioAdapterError(
                "Every native audio decoder layer must contain one gated MLP."
            )
        parent_parameters = {
            parameter.path: parameter for parameter in parent_nodes[0].parameters
        }
        child_parameters = {
            parameter.path: parameter for parameter in child_nodes[0].parameters
        }
        if set(parent_parameters) != set(child_parameters):
            raise DynamicAudioAdapterError(
                "Materialized native audio child changed gated-MLP parameter "
                "ownership."
            )
        suffix_shapes = {
            "gate_proj.weight": (
                (source_intermediate, hidden_size),
                (target_intermediate, hidden_size),
            ),
            "up_proj.weight": (
                (source_intermediate, hidden_size),
                (target_intermediate, hidden_size),
            ),
            "down_proj.weight": (
                (hidden_size, source_intermediate),
                (hidden_size, target_intermediate),
            ),
        }
        matched: set[str] = set()
        for path in parent_parameters:
            suffix = next(
                (value for value in suffix_shapes if path.endswith(value)),
                None,
            )
            if suffix is None:
                raise DynamicAudioAdapterError(
                    "Native audio gated MLP contains an unsupported learned "
                    "tensor."
                )
            expected_parent, expected_child = suffix_shapes[suffix]
            if (
                parent_parameters[path].shape != expected_parent
                or child_parameters[path].shape != expected_child
            ):
                raise DynamicAudioAdapterError(
                    f"Native audio MLP tensor {path!r} does not match the "
                    "requested width."
                )
            allowed_changed_paths.add(path)
            matched.add(suffix)
        if matched != set(suffix_shapes):
            raise DynamicAudioAdapterError(
                "Native audio gated MLP does not own one coupled "
                "gate/up/down triplet."
            )

    observed_changed = {
        path for path in parent_shapes if parent_shapes[path] != child_shapes[path]
    }
    if observed_changed != allowed_changed_paths:
        raise DynamicAudioAdapterError(
            "Materialized native audio child changed parameter shapes outside "
            "the coupled decoder MLP triplets."
        )
    expected_delta = (
        len(parent.layer_indices)
        * (source_intermediate - target_intermediate)
        * hidden_size
        * 3
    )
    if parent.unique_parameter_count - child.unique_parameter_count != expected_delta:
        raise DynamicAudioAdapterError(
            "Materialized native audio child parameter delta does not equal the "
            "exact coupled SwiGLU width reduction."
        )
    return expected_delta


class NativeAudioTargetSynthesisAdapter:
    """Clone one loaded native audio class/config and reduce only MLP width."""

    adapter_id = "neural_forge.audio.native_audio"
    version = 1
    priority = 240
    target_families = (NATIVE_AUDIO_FAMILY,)
    target_modalities = TARGET_MODALITIES

    def __init__(
        self,
        *,
        device: Any | None = None,
        dtype: Any | None = None,
    ) -> None:
        self.device = device
        self.dtype = dtype

    def supports(
        self,
        request: DynamicTargetRequest,
        parent_graphs: Mapping[str, ModelCircuitGraph],
    ) -> bool:
        return (
            request.target_family in {"auto", NATIVE_AUDIO_FAMILY}
            and request.target_modalities == TARGET_MODALITIES
            and any(
                graph.family == NATIVE_AUDIO_FAMILY
                for graph in parent_graphs.values()
            )
        )

    @staticmethod
    def _source_name(
        request: DynamicTargetRequest,
        parent_graphs: Mapping[str, ModelCircuitGraph],
    ) -> str:
        configuration = dict(request.configuration)
        configured = configuration.get("source_parent")
        if configured is not None:
            if not isinstance(configured, str) or configured not in parent_graphs:
                raise DynamicAudioAdapterError(
                    "Configured source_parent is not a loaded parent."
                )
            if parent_graphs[configured].family != NATIVE_AUDIO_FAMILY:
                raise DynamicAudioAdapterError(
                    "Configured source_parent is not a native audio graph."
                )
            return configured
        candidates = [
            (name, graph)
            for name, graph in parent_graphs.items()
            if graph.family == NATIVE_AUDIO_FAMILY
        ]
        if not candidates:
            raise DynamicAudioAdapterError(
                "No loaded native audio parent can define the target."
            )
        candidates.sort(key=lambda item: (-item[1].unique_parameter_count, item[0]))
        return candidates[0][0]

    @staticmethod
    def _requested_intermediate(request: DynamicTargetRequest) -> int:
        configuration = dict(request.configuration)
        unknown = set(configuration) - {
            "decoder_intermediate_size",
            "fold_cache_key",
            "source_parent",
        }
        if unknown:
            raise DynamicAudioAdapterError(
                "Unsupported native audio target configuration keys: "
                + ", ".join(sorted(unknown))
            )
        if "decoder_intermediate_size" not in configuration:
            raise DynamicAudioAdapterError(
                "Native audio target configuration requires "
                "decoder_intermediate_size."
            )
        fold_cache_key = configuration.get("fold_cache_key")
        if fold_cache_key is not None and (
            not isinstance(fold_cache_key, str)
            or len(fold_cache_key) != 64
            or any(character not in "0123456789abcdef" for character in fold_cache_key)
        ):
            raise DynamicAudioAdapterError(
                "fold_cache_key must be a lowercase SHA-256 digest."
            )
        return _positive_int(
            configuration["decoder_intermediate_size"],
            "decoder_intermediate_size",
        )

    def materialize(
        self,
        request: DynamicTargetRequest,
        parent_models: Mapping[str, Any],
        parent_graphs: Mapping[str, ModelCircuitGraph],
        inspectors: Any,
    ) -> AdapterMaterialization:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional torch boundary
            raise DynamicAudioAdapterError(
                "PyTorch is required for native audio target synthesis."
            ) from exc

        source_name = self._source_name(request, parent_graphs)
        target_intermediate = self._requested_intermediate(request)
        parent = parent_models[source_name]
        parent_graph = parent_graphs[source_name]
        parent_config = getattr(parent, "config", None)
        decoder_config = getattr(parent_config, "decoder_config", None)
        audio_config = getattr(parent_config, "audio_config", None)
        if (
            parent_config is None
            or getattr(parent_config, "model_type", None) != NATIVE_AUDIO_MODEL_TYPE
            or getattr(parent_config, NATIVE_AUDIO_LANE_CONFIG_KEY, None)
            != NATIVE_AUDIO_LANE_MARKER
            or decoder_config is None
            or audio_config is None
        ):
            raise DynamicAudioAdapterError(
                "Native audio target synthesis requires one loaded local "
                "native_audio_chat parent class and marked config."
            )
        source_intermediate = _positive_int(
            getattr(decoder_config, "intermediate_size", None),
            "parent decoder intermediate_size",
        )
        if target_intermediate >= source_intermediate:
            raise DynamicAudioAdapterError(
                "decoder_intermediate_size must be smaller than the parent width."
            )
        hidden_size = _positive_int(
            getattr(decoder_config, "hidden_size", None),
            "parent decoder hidden_size",
        )
        if request.tokenizer_fingerprint != parent_graph.tokenizer_fingerprint:
            raise DynamicAudioAdapterError(
                "Requested child tokenizer differs from the selected native "
                "audio source."
            )

        parameters = tuple(parent.parameters())
        if not parameters:
            raise DynamicAudioAdapterError(
                "Loaded native audio parent has no parameters."
            )
        parent_devices = {parameter.device for parameter in parameters}
        parent_dtypes = {parameter.dtype for parameter in parameters}
        if self.device is None:
            if len(parent_devices) != 1:
                raise DynamicAudioAdapterError(
                    "A sharded native audio parent requires an explicit child "
                    "device."
                )
            device = next(iter(parent_devices))
        else:
            device = torch.device(self.device)
        if self.dtype is None:
            if len(parent_dtypes) != 1:
                raise DynamicAudioAdapterError(
                    "A mixed-dtype native audio parent requires an explicit "
                    "child dtype."
                )
            dtype = next(iter(parent_dtypes))
        else:
            dtype = self.dtype
            if len(parent_dtypes) != 1 or dtype != next(iter(parent_dtypes)):
                raise DynamicAudioAdapterError(
                    "Changing parameter dtype is outside the native audio "
                    "intermediate-size-only target contract."
                )

        source_payload = _config_payload(parent_config)
        target_config = copy.deepcopy(parent_config)
        target_config.decoder_config.intermediate_size = target_intermediate
        target_payload = _config_payload(target_config)
        differences = _leaf_differences(source_payload, target_payload)
        if differences != (("decoder_config", "intermediate_size"),):
            raise DynamicAudioAdapterError(
                "Native audio target config changed fields other than "
                "decoder_config.intermediate_size."
            )

        cuda_devices: list[int] = []
        if device.type == "cuda":
            index = (
                torch.cuda.current_device() if device.index is None else device.index
            )
            cuda_devices = [index]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(request.blank_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(request.blank_seed)
            with _temporary_default_dtype(torch, dtype), torch.device(device):
                child = type(parent)(copy.deepcopy(target_config))
        child.to(device=device)
        child.eval()
        if any(
            parameter.device != device or parameter.dtype != dtype
            for parameter in child.parameters()
        ):
            raise DynamicAudioAdapterError(
                "Direct native audio blank construction missed the parent "
                "device/dtype."
            )
        if any(parameter.grad is not None for parameter in child.parameters()):
            raise DynamicAudioAdapterError(
                "Fresh native audio blank unexpectedly contains gradients."
            )
        child_payload = _config_payload(getattr(child, "config", None))
        child_differences = _leaf_differences(source_payload, child_payload)
        if child_differences != (("decoder_config", "intermediate_size"),):
            raise DynamicAudioAdapterError(
                "Local native audio class construction mutated configuration "
                "outside the requested decoder MLP width."
            )

        graph = inspectors.inspect(
            child,
            model_name=request.child_name,
            tokenizer_fingerprint=request.tokenizer_fingerprint,
        )
        if graph.family != NATIVE_AUDIO_FAMILY:
            raise DynamicAudioAdapterError(
                "Independent inspection did not identify a native audio child."
            )
        modalities = tuple(sorted({value.modality for value in graph.interfaces}))
        if modalities != request.target_modalities:
            raise DynamicAudioAdapterError(
                "Independent native audio child modalities differ from the "
                "request."
            )
        expected_delta = _validate_only_decoder_mlp_width_changed(
            parent_graph,
            graph,
            source_intermediate=source_intermediate,
            target_intermediate=target_intermediate,
            hidden_size=hidden_size,
        )
        if graph.unique_parameter_count != request.exact_unique_parameters:
            raise DynamicAudioAdapterError(
                "Materialized native audio child misses the exact requested "
                "parameter count."
            )
        if (
            request.maximum_parameter_bytes is not None
            and graph.unique_parameter_bytes > request.maximum_parameter_bytes
        ):
            raise DynamicAudioAdapterError(
                "Materialized native audio child exceeds the requested "
                "parameter-byte cap."
            )

        topology_payload = _graph_topology_payload(graph)
        topology_fingerprint = canonical_sha256(
            {
                "schema": ADAPTER_SCHEMA,
                "source_parent": source_name,
                "source_graph": parent_graph.fingerprint,
                "child_topology": topology_payload,
                "source_intermediate_size": source_intermediate,
                "target_intermediate_size": target_intermediate,
                "config_change": "decoder_config.intermediate_size",
                "exact_parameter_delta": expected_delta,
            }
        )
        return AdapterMaterialization(
            model=child,
            graph=graph,
            topology_fingerprint=topology_fingerprint,
            report={
                "source_parent": source_name,
                "source_intermediate_size": source_intermediate,
                "target_intermediate_size": target_intermediate,
                "decoder_layer_count": len(graph.layer_indices),
                "decoder_hidden_size": hidden_size,
                "audio_front_end_copied_exactly": True,
                "exact_parameter_delta": expected_delta,
                "blank_seed": request.blank_seed,
                "config_fields_changed": "decoder_config.intermediate_size",
                "local_parent_class_instantiation": True,
                "direct_target_device_construction": True,
                "hub_loader_used": False,
                "optimizer_steps": 0,
                "backward_calls": 0,
            },
            adapter_payload={
                "schema": ADAPTER_SCHEMA,
                "source_parent": source_name,
                "source_graph_fingerprint": parent_graph.fingerprint,
                "target_config_fingerprint": canonical_sha256(child_payload),
                "topology_fingerprint": topology_fingerprint,
            },
        )


__all__ = [
    "ADAPTER_SCHEMA",
    "TARGET_MODALITIES",
    "DynamicAudioAdapterError",
    "NativeAudioTargetSynthesisAdapter",
]
