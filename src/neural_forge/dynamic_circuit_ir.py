"""Architecture-neutral causal-circuit intermediate representation.

The dynamic folding compiler must reason about functions, not filename
conventions or numerically similar tensors.  This module describes a model as
ordered causal circuits with explicit residual, state/cache, token, and
modality edges.  It contains no framework objects and is safe to serialize,
fingerprint, compare, and hand to family-specific folding operators.

The module also owns the canonical modality-name registry
(:data:`CANONICAL_MODALITIES`).  Modality strings on ports and interfaces stay
free-form so every existing graph and fingerprint keeps validating, but new
lanes validate their names through :func:`canonical_modality` so that the IR,
the raw-byte codecs, and the behavior gates cannot drift into private
spellings of the same modality.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
SCHEMA = "neural-forge-dynamic-circuit-ir-v1"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

Dimension = int | str
JsonScalar = str | int | float | bool | None


class DynamicCircuitIRError(ValueError):
    """The graph is incomplete, ambiguous, or internally inconsistent."""


class CircuitKind(str, Enum):
    TOKEN_EMBEDDING = "token_embedding"
    POSITION_ENCODING = "position_encoding"
    MODALITY_INPUT = "modality_input"
    MODALITY_OUTPUT = "modality_output"
    NORM = "norm"
    ATTENTION = "attention"
    SHORT_CONV = "short_conv"
    RECURRENT_MIXER = "recurrent_mixer"
    DENSE_MLP = "dense_mlp"
    GATED_MLP = "gated_mlp"
    RESIDUAL_ADD = "residual_add"
    OUTPUT_HEAD = "output_head"
    INTERFACE_PROJECTION = "interface_projection"


class EdgeKind(str, Enum):
    DATA = "data"
    RESIDUAL = "residual"
    CONTROL = "control"
    TOKEN = "token"
    STATE = "state"
    CACHE = "cache"
    MODALITY = "modality"


class PortKind(str, Enum):
    TOKEN_IDS = "token_ids"
    POSITION_IDS = "position_ids"
    RESIDUAL = "residual"
    NORMALIZED_RESIDUAL = "normalized_residual"
    BRANCH_DELTA = "branch_delta"
    LOGITS = "logits"
    ATTENTION_MASK = "attention_mask"
    CACHE = "cache"
    RECURRENT_STATE = "recurrent_state"
    MODALITY = "modality"
    GENERIC = "generic"


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise DynamicCircuitIRError("IR payload is not canonical JSON.") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value.strip()):
        raise DynamicCircuitIRError(
            f"{label} must be a nonempty stable identifier containing only "
            "letters, digits, '.', '_', ':', '/', or '-'."
        )
    return value.strip()


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DynamicCircuitIRError(f"{label} must be a nonempty string.")
    return value.strip()


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DynamicCircuitIRError(f"{label} must be a non-negative integer.")
    return value


def _positive_int(value: Any, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result == 0:
        raise DynamicCircuitIRError(f"{label} must be positive.")
    return result


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise DynamicCircuitIRError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _scalar(value: Any, label: str) -> JsonScalar:
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise DynamicCircuitIRError(f"{label} must be a JSON scalar.")
    if isinstance(value, float) and not math.isfinite(value):
        raise DynamicCircuitIRError(f"{label} must be finite.")
    return value


def _attributes(
    value: Mapping[str, JsonScalar] | Iterable[tuple[str, JsonScalar]] | None,
    label: str,
) -> tuple[tuple[str, JsonScalar], ...]:
    if value is None:
        return ()
    items = value.items() if isinstance(value, Mapping) else value
    result = []
    for key, item in items:
        result.append((_name(key, f"{label} key"), _scalar(item, f"{label}.{key}")))
    result.sort(key=lambda pair: pair[0])
    if len({key for key, _ in result}) != len(result):
        raise DynamicCircuitIRError(f"{label} keys must be unique.")
    return tuple(result)


def _dimension(value: Any, label: str) -> Dimension:
    if isinstance(value, bool):
        raise DynamicCircuitIRError(f"{label} is not a valid dimension.")
    if isinstance(value, int):
        return _positive_int(value, label)
    return _name(value, label)


MODALITY_REGISTRY_SCHEMA = "neural-forge-canonical-modality-registry-v1"

MODALITY_TEXT = "text"
MODALITY_IMAGE = "image"
MODALITY_AUDIO = "audio"
MODALITY_VIDEO = "video"
MODALITY_OCR = "ocr"
MODALITY_POINTS3D = "points3d"

CANONICAL_MODALITY_ORDER: tuple[str, ...] = (
    MODALITY_TEXT,
    MODALITY_IMAGE,
    MODALITY_AUDIO,
    MODALITY_VIDEO,
    MODALITY_OCR,
    MODALITY_POINTS3D,
)
CANONICAL_MODALITIES: frozenset[str] = frozenset(CANONICAL_MODALITY_ORDER)

_MODALITY_NOTES: Mapping[str, str] = MappingProxyType(
    {
        MODALITY_TEXT: (
            "Ordinary tokenized or byte-level text; the default parent and "
            "child interface of every folding lane."
        ),
        MODALITY_IMAGE: (
            "Decoded raster imagery.  Content binding is the decoded-RGB "
            "digest, not the container digest."
        ),
        MODALITY_AUDIO: (
            "Decoded waveform audio.  Content binding is the decoded PCM16 "
            "digest, not the container digest."
        ),
        MODALITY_VIDEO: (
            "Decoded frame sequences.  No lane emits this yet; reserved so "
            "that a later lane cannot invent a divergent spelling."
        ),
        MODALITY_OCR: (
            "Rendered-text imagery carried as its own transport modality by "
            "the raw-byte codecs.  It is NOT the NEO 'ocr' probe category, "
            "which is a text-answer task category rather than a modality."
        ),
        MODALITY_POINTS3D: (
            "Point-cloud style geometry carried by the raw-byte codecs.  No "
            "inspector emits this yet."
        ),
    }
)


def is_canonical_modality(value: Any) -> bool:
    """Return whether ``value`` is exactly one canonical modality name."""

    return isinstance(value, str) and value.strip() in CANONICAL_MODALITIES


def canonical_modality(value: Any, label: str = "modality") -> str:
    """Validate one canonical modality name and return its stripped form.

    Modality strings inside :class:`CircuitPort` and
    :class:`ModalityInterface` deliberately stay free-form so that existing
    graphs, receipts, and fingerprints keep validating unchanged.  New lanes
    opt into this helper instead of inventing a private spelling.
    """

    name = _name(value, label)
    if name not in CANONICAL_MODALITIES:
        raise DynamicCircuitIRError(
            f"{label} must be one canonical modality name: "
            + ", ".join(CANONICAL_MODALITY_ORDER)
            + f"; received {name!r}."
        )
    return name


def modality_registry() -> dict[str, Any]:
    """Return the canonical modality registry as a JSON-safe statement."""

    return {
        "schema": MODALITY_REGISTRY_SCHEMA,
        "modalities": list(CANONICAL_MODALITY_ORDER),
        "notes": dict(_MODALITY_NOTES),
        "ir_strings_remain_free_form": True,
        "new_lanes_must_use_canonical_names": True,
        "validation_helper": "neural_forge.dynamic_circuit_ir.canonical_modality",
        "task_categories_are_not_modalities": True,
    }


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """One named learned tensor and its canonical functional role."""

    path: str
    role: str
    shape: tuple[int, ...]
    dtype: str
    storage_group: str
    trainable: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _name(self.path, "parameter path"))
        object.__setattr__(self, "role", _name(self.role, "parameter role"))
        shape = tuple(
            _positive_int(value, "parameter dimension") for value in self.shape
        )
        if not shape:
            raise DynamicCircuitIRError("Parameter shape cannot be scalar or empty.")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", _text(self.dtype, "parameter dtype"))
        object.__setattr__(
            self,
            "storage_group",
            _name(self.storage_group, "parameter storage_group"),
        )
        if not isinstance(self.trainable, bool):
            raise DynamicCircuitIRError("parameter trainable must be boolean.")

    @property
    def numel(self) -> int:
        result = 1
        for value in self.shape:
            result *= value
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "role": self.role,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "storage_group": self.storage_group,
            "trainable": self.trainable,
            "numel": self.numel,
        }


@dataclass(frozen=True, slots=True)
class CircuitPort:
    """One typed input or output of a causal circuit."""

    name: str
    kind: PortKind | str
    dimensions: tuple[Dimension, ...] = ()
    modality: str | None = None
    stateful: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "port name"))
        kind = self.kind.value if isinstance(self.kind, PortKind) else self.kind
        object.__setattr__(self, "kind", _name(kind, "port kind"))
        object.__setattr__(
            self,
            "dimensions",
            tuple(
                _dimension(value, f"port {self.name} dimension")
                for value in self.dimensions
            ),
        )
        if self.modality is not None:
            object.__setattr__(self, "modality", _name(self.modality, "port modality"))
        if not isinstance(self.stateful, bool):
            raise DynamicCircuitIRError("port stateful must be boolean.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "dimensions": list(self.dimensions),
            "modality": self.modality,
            "stateful": self.stateful,
        }


@dataclass(frozen=True, slots=True)
class CircuitNode:
    """One indivisible functional unit for planning and folding."""

    node_id: str
    kind: CircuitKind | str
    order: int
    path: str
    layer_index: int | None
    inputs: tuple[CircuitPort, ...]
    outputs: tuple[CircuitPort, ...]
    parameters: tuple[ParameterSpec, ...] = ()
    attributes: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _name(self.node_id, "node_id"))
        kind = self.kind.value if isinstance(self.kind, CircuitKind) else self.kind
        object.__setattr__(self, "kind", _name(kind, "circuit kind"))
        object.__setattr__(self, "order", _nonnegative_int(self.order, "node order"))
        object.__setattr__(self, "path", _name(self.path, "node path"))
        if self.layer_index is not None:
            object.__setattr__(
                self,
                "layer_index",
                _nonnegative_int(self.layer_index, "layer_index"),
            )
        inputs = tuple(self.inputs)
        outputs = tuple(self.outputs)
        if any(not isinstance(value, CircuitPort) for value in (*inputs, *outputs)):
            raise DynamicCircuitIRError("Node ports must be CircuitPort objects.")
        if len({value.name for value in inputs}) != len(inputs):
            raise DynamicCircuitIRError(f"Node {self.node_id!r} input names repeat.")
        if len({value.name for value in outputs}) != len(outputs):
            raise DynamicCircuitIRError(f"Node {self.node_id!r} output names repeat.")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)
        parameters = tuple(self.parameters)
        if any(not isinstance(value, ParameterSpec) for value in parameters):
            raise DynamicCircuitIRError(
                "Node parameters must be ParameterSpec objects."
            )
        if len({value.path for value in parameters}) != len(parameters):
            raise DynamicCircuitIRError(
                f"Node {self.node_id!r} contains duplicate parameter paths."
            )
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(
            self,
            "attributes",
            _attributes(self.attributes, f"node {self.node_id} attributes"),
        )

    @property
    def parameter_paths(self) -> tuple[str, ...]:
        return tuple(value.path for value in self.parameters)

    @property
    def logical_parameter_count(self) -> int:
        return sum(value.numel for value in self.parameters)

    def input(self, name: str) -> CircuitPort:
        for value in self.inputs:
            if value.name == name:
                return value
        raise KeyError(name)

    def output(self, name: str) -> CircuitPort:
        for value in self.outputs:
            if value.name == name:
                return value
        raise KeyError(name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "order": self.order,
            "path": self.path,
            "layer_index": self.layer_index,
            "inputs": [value.as_dict() for value in self.inputs],
            "outputs": [value.as_dict() for value in self.outputs],
            "parameters": [value.as_dict() for value in self.parameters],
            "attributes": dict(self.attributes),
            "logical_parameter_count": self.logical_parameter_count,
        }


@dataclass(frozen=True, slots=True)
class CircuitEdge:
    """One explicit causal, residual, state, cache, token, or modality edge."""

    source_node: str
    source_port: str
    target_node: str
    target_port: str
    kind: EdgeKind | str
    recurrent: bool = False
    attributes: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        for name in ("source_node", "source_port", "target_node", "target_port"):
            object.__setattr__(self, name, _name(getattr(self, name), name))
        kind = self.kind.value if isinstance(self.kind, EdgeKind) else self.kind
        object.__setattr__(self, "kind", _name(kind, "edge kind"))
        if not isinstance(self.recurrent, bool):
            raise DynamicCircuitIRError("edge recurrent must be boolean.")
        object.__setattr__(
            self,
            "attributes",
            _attributes(self.attributes, "edge attributes"),
        )

    @property
    def edge_id(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_node": self.source_node,
            "source_port": self.source_port,
            "target_node": self.target_node,
            "target_port": self.target_port,
            "kind": self.kind,
            "recurrent": self.recurrent,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True, slots=True)
class ModalityInterface:
    """A model boundary exposed to a parent, child, tokenizer, or raw codec."""

    interface_id: str
    modality: str
    direction: str
    node_id: str
    port_name: str
    representation: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "interface_id", _name(self.interface_id, "interface_id")
        )
        object.__setattr__(self, "modality", _name(self.modality, "modality"))
        direction = _name(self.direction, "interface direction")
        if direction not in {"input", "output"}:
            raise DynamicCircuitIRError(
                "interface direction must be 'input' or 'output'."
            )
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "node_id", _name(self.node_id, "interface node_id"))
        object.__setattr__(
            self, "port_name", _name(self.port_name, "interface port_name")
        )
        object.__setattr__(
            self,
            "representation",
            _name(self.representation, "interface representation"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "interface_id": self.interface_id,
            "modality": self.modality,
            "direction": self.direction,
            "node_id": self.node_id,
            "port_name": self.port_name,
            "representation": self.representation,
        }


@dataclass(frozen=True, slots=True)
class ModelCircuitGraph:
    """Complete immutable causal-circuit graph for one concrete model."""

    model_name: str
    family: str
    model_class: str
    nodes: tuple[CircuitNode, ...]
    edges: tuple[CircuitEdge, ...]
    interfaces: tuple[ModalityInterface, ...]
    logical_parameter_count: int
    unique_parameter_count: int
    unique_parameter_bytes: int
    config_fingerprint: str
    tokenizer_fingerprint: str | None = None
    metadata: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_name", _name(self.model_name, "model_name"))
        object.__setattr__(self, "family", _name(self.family, "family"))
        object.__setattr__(self, "model_class", _text(self.model_class, "model_class"))
        nodes = tuple(
            sorted(self.nodes, key=lambda value: (value.order, value.node_id))
        )
        if not nodes or any(not isinstance(value, CircuitNode) for value in nodes):
            raise DynamicCircuitIRError("A model graph needs CircuitNode objects.")
        if len({value.node_id for value in nodes}) != len(nodes):
            raise DynamicCircuitIRError("Model graph node IDs must be unique.")
        object.__setattr__(self, "nodes", nodes)
        edges = tuple(
            sorted(
                self.edges,
                key=lambda value: (
                    value.source_node,
                    value.source_port,
                    value.target_node,
                    value.target_port,
                    value.kind,
                ),
            )
        )
        if any(not isinstance(value, CircuitEdge) for value in edges):
            raise DynamicCircuitIRError(
                "Model graph edges must be CircuitEdge objects."
            )
        if len({value.edge_id for value in edges}) != len(edges):
            raise DynamicCircuitIRError("Model graph edges must be unique.")
        object.__setattr__(self, "edges", edges)
        interfaces = tuple(
            sorted(self.interfaces, key=lambda value: value.interface_id)
        )
        if any(not isinstance(value, ModalityInterface) for value in interfaces):
            raise DynamicCircuitIRError(
                "Model graph interfaces must be ModalityInterface objects."
            )
        if len({value.interface_id for value in interfaces}) != len(interfaces):
            raise DynamicCircuitIRError("Model interface IDs must be unique.")
        object.__setattr__(self, "interfaces", interfaces)
        for name in (
            "logical_parameter_count",
            "unique_parameter_count",
            "unique_parameter_bytes",
        ):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        if self.unique_parameter_count > self.logical_parameter_count:
            raise DynamicCircuitIRError(
                "unique_parameter_count cannot exceed logical_parameter_count."
            )
        object.__setattr__(
            self,
            "config_fingerprint",
            _sha256(self.config_fingerprint, "config_fingerprint"),
        )
        if self.tokenizer_fingerprint is not None:
            object.__setattr__(
                self,
                "tokenizer_fingerprint",
                _sha256(self.tokenizer_fingerprint, "tokenizer_fingerprint"),
            )
        object.__setattr__(
            self, "metadata", _attributes(self.metadata, "model metadata")
        )
        self._validate_references()
        captured = [path for node in nodes for path in node.parameter_paths]
        if len(set(captured)) != len(captured):
            raise DynamicCircuitIRError(
                "Each named parameter path must belong to exactly one circuit node."
            )
        if sum(node.logical_parameter_count for node in nodes) != (
            self.logical_parameter_count
        ):
            raise DynamicCircuitIRError(
                "Circuit-node parameters do not exactly cover logical parameters."
            )

    def _validate_references(self) -> None:
        by_id = {value.node_id: value for value in self.nodes}
        for edge in self.edges:
            if edge.source_node not in by_id or edge.target_node not in by_id:
                raise DynamicCircuitIRError("An edge references an unknown node.")
            source = by_id[edge.source_node]
            target = by_id[edge.target_node]
            try:
                source_port = source.output(edge.source_port)
                target_port = target.input(edge.target_port)
            except KeyError as exc:
                raise DynamicCircuitIRError(
                    "An edge references an unknown source output or target input."
                ) from exc
            if not edge.recurrent and source.order > target.order:
                raise DynamicCircuitIRError(
                    "A non-recurrent edge cannot point backward in causal order."
                )
            if source_port.kind != target_port.kind:
                raise DynamicCircuitIRError(
                    "An edge connects incompatible source and target port kinds."
                )
            if source_port.dimensions != target_port.dimensions:
                raise DynamicCircuitIRError(
                    "An edge connects incompatible source and target dimensions."
                )
            if (
                source_port.modality is not None
                and target_port.modality is not None
                and source_port.modality != target_port.modality
            ):
                raise DynamicCircuitIRError("An edge connects incompatible modalities.")
            if edge.kind in {EdgeKind.STATE.value, EdgeKind.CACHE.value}:
                if not source_port.stateful or not target_port.stateful:
                    raise DynamicCircuitIRError(
                        "State/cache edges require stateful ports."
                    )
                if not edge.recurrent:
                    raise DynamicCircuitIRError(
                        "State/cache edges must be explicitly recurrent."
                    )
            elif edge.recurrent:
                raise DynamicCircuitIRError("Only state/cache edges may be recurrent.")
        indegree = {node_id: 0 for node_id in by_id}
        outgoing: dict[str, list[str]] = {node_id: [] for node_id in by_id}
        for edge in self.edges:
            if edge.recurrent:
                continue
            outgoing[edge.source_node].append(edge.target_node)
            indegree[edge.target_node] += 1
        ready = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
        visited = 0
        while ready:
            node_id = ready.pop(0)
            visited += 1
            for target_id in sorted(outgoing[node_id]):
                indegree[target_id] -= 1
                if indegree[target_id] == 0:
                    ready.append(target_id)
                    ready.sort()
        if visited != len(by_id):
            raise DynamicCircuitIRError(
                "Non-recurrent circuit edges contain a causal cycle."
            )
        for interface in self.interfaces:
            if interface.node_id not in by_id:
                raise DynamicCircuitIRError("An interface references an unknown node.")
            node = by_id[interface.node_id]
            ports = node.inputs if interface.direction == "input" else node.outputs
            if interface.port_name not in {value.name for value in ports}:
                raise DynamicCircuitIRError(
                    "An interface references an unknown boundary port."
                )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict(include_fingerprint=False))

    @property
    def structural_topology_fingerprint(self) -> str:
        """Hash topology while excluding parameter execution policy only."""

        return canonical_sha256(self.as_structural_topology_dict())

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    node.layer_index
                    for node in self.nodes
                    if node.layer_index is not None
                }
            )
        )

    @property
    def parameter_paths(self) -> tuple[str, ...]:
        return tuple(path for node in self.nodes for path in node.parameter_paths)

    def node(self, node_id: str) -> CircuitNode:
        for value in self.nodes:
            if value.node_id == node_id:
                return value
        raise KeyError(node_id)

    def nodes_of_kind(
        self,
        kind: CircuitKind | str,
        *,
        layer_index: int | None = None,
    ) -> tuple[CircuitNode, ...]:
        normalized = kind.value if isinstance(kind, CircuitKind) else str(kind)
        return tuple(
            value
            for value in self.nodes
            if value.kind == normalized
            and (layer_index is None or value.layer_index == layer_index)
        )

    def layer_nodes(self, layer_index: int) -> tuple[CircuitNode, ...]:
        index = _nonnegative_int(layer_index, "layer_index")
        return tuple(value for value in self.nodes if value.layer_index == index)

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "model_name": self.model_name,
            "family": self.family,
            "model_class": self.model_class,
            "nodes": [value.as_dict() for value in self.nodes],
            "edges": [value.as_dict() for value in self.edges],
            "interfaces": [value.as_dict() for value in self.interfaces],
            "logical_parameter_count": self.logical_parameter_count,
            "unique_parameter_count": self.unique_parameter_count,
            "unique_parameter_bytes": self.unique_parameter_bytes,
            "config_fingerprint": self.config_fingerprint,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "metadata": dict(self.metadata),
            "complete_parameter_coverage": True,
            "explicit_residual_edges": any(
                value.kind == EdgeKind.RESIDUAL.value for value in self.edges
            ),
            "explicit_state_or_cache_edges": any(
                value.kind in {EdgeKind.STATE.value, EdgeKind.CACHE.value}
                for value in self.edges
            ),
        }
        if include_fingerprint:
            payload["fingerprint"] = canonical_sha256(payload)
        return payload

    def as_structural_topology_dict(self) -> dict[str, Any]:
        """Return the full canonical graph minus ``ParameterSpec.trainable``.

        ``trainable`` mirrors a runtime ``requires_grad`` execution policy.  It
        is intentionally excluded from structural identity, while every other
        graph leaf -- including parameter paths, roles, shapes, dtypes, storage
        alias groups, configuration, tokenization, and graph metadata -- stays
        in the canonical payload.
        """

        payload = self.as_dict(include_fingerprint=False)
        for node in payload["nodes"]:
            for parameter in node["parameters"]:
                del parameter["trainable"]
        return payload


__all__ = [
    "CANONICAL_MODALITIES",
    "CANONICAL_MODALITY_ORDER",
    "CircuitEdge",
    "CircuitKind",
    "CircuitNode",
    "CircuitPort",
    "Dimension",
    "DynamicCircuitIRError",
    "EdgeKind",
    "JsonScalar",
    "MODALITY_AUDIO",
    "MODALITY_IMAGE",
    "MODALITY_OCR",
    "MODALITY_POINTS3D",
    "MODALITY_REGISTRY_SCHEMA",
    "MODALITY_TEXT",
    "MODALITY_VIDEO",
    "ModalityInterface",
    "ModelCircuitGraph",
    "ParameterSpec",
    "PortKind",
    "SCHEMA",
    "SCHEMA_VERSION",
    "canonical_json_bytes",
    "canonical_modality",
    "canonical_sha256",
    "is_canonical_modality",
    "modality_registry",
]
