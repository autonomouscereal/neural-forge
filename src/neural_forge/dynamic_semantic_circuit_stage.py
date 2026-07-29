"""Storage-authenticated semantic circuit-stage compilation ABI.

This module orders analytic component compilers *inside* the existing outer
whole-model semantic transaction.  It observes and authenticates writes, but
never snapshots values for restoration, rolls the child back, decides model
acceptance, or publishes an artifact.  A failure therefore leaves the child
exactly where the failing callback left it so the outer transaction remains
the single rollback owner.

Contracts contain no parameter paths.  They select inspected circuit kinds
and pluggable parameter semantics.  Before any callback runs, the registry:

* authenticates the live parameter shape, dtype, object, view, and storage
  alias topology against :class:`ModelCircuitGraph`;
* resolves selectors from the authenticated graph and closes every selection
  over all paths sharing a graph ``storage_group``;
* proves one complete dependency DAG;
* proves an explicit, ordered handoff chain for every multiply-written storage
  group; and
* proves exact, disjoint final receipt ownership of every learned storage
  group.

Around every callback the registry hashes every learned path and underlying
storage.  It also rehashes the callback-visible graph, manifest, resolved
stages, frozen registry, and prior result bodies against private pre-callback
seals.  It rejects writes outside the preflighted access surface, then
constructs the immutable result itself so a compiler cannot self-assert which
parameters changed or rewrite its authorization metadata.

The current enforcement scope is learned tensors returned by
``named_parameters(remove_duplicate=False)``.  Persistent/non-persistent
buffers and architectures that expose distinct partial views of one physical
storage are not claimed by this ABI version; those require a later IR extension
rather than an implicit coverage claim here.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from .dynamic_behavior_gate import (
    BehaviorCallableIdentity,
    identify_behavior_callable,
)
from .dynamic_circuit_ir import (
    CircuitKind,
    ModelCircuitGraph,
    ParameterSpec,
    canonical_sha256,
)
from .dynamic_closed_loop_fold import CompileStage
from .dynamic_fold_budget import (
    SEMANTIC_ATTENTION_QUERY_KEY,
    SEMANTIC_ATTENTION_WRITER,
    SEMANTIC_INTERFACE,
    SEMANTIC_MLP_PRODUCER,
    SEMANTIC_MLP_WRITER,
    SEMANTIC_NORMALIZATION,
    SEMANTIC_OTHER,
    SEMANTIC_STATE_MIXER_INTERNAL,
    classify_component_parameter_semantic,
)


SCHEMA = "neural-forge-semantic-circuit-stage-v2"
DEFAULT_SEMANTIC_CLASSIFIER_RESOURCE_IDENTITY = canonical_sha256(
    {
        "schema": SCHEMA,
        "kind": "default_semantic_classifier_resource",
        "policy": "canonical-kind-role-with-fused-role-union-v1",
    }
)

_STABLE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*\Z")
_FUSED_ROLE_SEPARATOR = re.compile(r"[+,|;/]+")
_KNOWN_SEMANTICS = frozenset(
    {
        SEMANTIC_STATE_MIXER_INTERNAL,
        SEMANTIC_ATTENTION_QUERY_KEY,
        SEMANTIC_ATTENTION_WRITER,
        SEMANTIC_MLP_PRODUCER,
        SEMANTIC_MLP_WRITER,
        SEMANTIC_NORMALIZATION,
        SEMANTIC_INTERFACE,
        SEMANTIC_OTHER,
    }
)


class DynamicSemanticCircuitStageError(ValueError):
    """A stage contract, live topology, callback, or observed write is invalid."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DynamicSemanticCircuitStageError(
            f"{name} must be a nonempty string."
        )
    return value.strip()


def _stable_name(value: Any, name: str) -> str:
    result = _text(value, name)
    if not _STABLE_NAME.fullmatch(result):
        raise DynamicSemanticCircuitStageError(
            f"{name} must be a stable identifier."
        )
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DynamicSemanticCircuitStageError(
            f"{name} must be a positive integer."
        )
    return value


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DynamicSemanticCircuitStageError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _unique_names(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(_stable_name(value, name) for value in values)
    if len(set(result)) != len(result):
        raise DynamicSemanticCircuitStageError(f"{name} values must be unique.")
    return tuple(sorted(result))


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json_object(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DynamicSemanticCircuitStageError(f"{name} must be a mapping.")
    try:
        encoded = json.dumps(
            _thaw_json(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise DynamicSemanticCircuitStageError(
            f"{name} must be canonical JSON."
        ) from exc
    if not isinstance(decoded, dict):
        raise DynamicSemanticCircuitStageError(
            f"{name} must encode a JSON object."
        )
    return _freeze_json(decoded)


def _normalize_dtype(value: Any) -> str:
    result = _text(str(value), "parameter dtype").casefold()
    return result.removeprefix("torch.")


def default_semantic_parameter_classifier(
    circuit_kind: str,
    parameter_role: str,
) -> tuple[str, ...]:
    """Return all canonical semantics represented by one inspected tensor.

    Most inspected roles describe one operation.  A fused tensor can describe
    several roles separated by ``+``, ``,`, ``|``, ``;``, or ``/``.  Each
    fragment is classified independently and the union is returned, allowing
    selectors to discover fused QK/writer or producer/writer tensors without
    consulting their paths.
    """

    kind = _stable_name(circuit_kind, "circuit_kind")
    role = _stable_name(parameter_role, "parameter_role")
    fragments = tuple(
        fragment.strip("._:-")
        for fragment in _FUSED_ROLE_SEPARATOR.split(role)
        if fragment.strip("._:-")
    )
    if not fragments:
        fragments = (role,)
    semantics = {
        classify_component_parameter_semantic(kind, fragment)
        for fragment in fragments
    }
    semantics.add(classify_component_parameter_semantic(kind, role))
    return tuple(sorted(semantics))


class SemanticParameterClassifier(Protocol):
    """Pluggable, explicitly identified functional-role classifier."""

    semantic_classifier_resource_identity: str

    def __call__(
        self,
        circuit_kind: str,
        parameter_role: str,
    ) -> Sequence[str]: ...


class _DefaultSemanticParameterClassifier:
    __slots__ = ()

    @property
    def semantic_classifier_resource_identity(self) -> str:
        return DEFAULT_SEMANTIC_CLASSIFIER_RESOURCE_IDENTITY

    def __call__(
        self,
        circuit_kind: str,
        parameter_role: str,
    ) -> tuple[str, ...]:
        return default_semantic_parameter_classifier(circuit_kind, parameter_role)


@dataclass(frozen=True, slots=True)
class SemanticCircuitSelector:
    """Path-free selector over authenticated circuit kinds and semantics.

    Values within an axis are alternatives and the two axes are conjunctive.
    Empty axes are wildcards, though both cannot be empty.  ``optional`` allows
    a registered circuit/semantic to be absent from a particular family while
    still keeping one reusable cross-family stage contract.
    """

    circuit_kinds: tuple[CircuitKind | str, ...] = ()
    parameter_semantics: tuple[str, ...] = ()
    optional: bool = False

    def __post_init__(self) -> None:
        kinds = tuple(
            sorted(
                {
                    value.value
                    if isinstance(value, CircuitKind)
                    else _stable_name(value, "selector circuit kind")
                    for value in self.circuit_kinds
                }
            )
        )
        semantics = tuple(
            sorted(
                {
                    _stable_name(value, "selector parameter semantic").lower()
                    for value in self.parameter_semantics
                }
            )
        )
        if len(kinds) != len(self.circuit_kinds):
            raise DynamicSemanticCircuitStageError(
                "Selector circuit kinds must be unique."
            )
        if len(semantics) != len(self.parameter_semantics):
            raise DynamicSemanticCircuitStageError(
                "Selector parameter semantics must be unique."
            )
        if not kinds and not semantics:
            raise DynamicSemanticCircuitStageError(
                "A semantic circuit selector needs a circuit kind or parameter "
                "semantic."
            )
        if not isinstance(self.optional, bool):
            raise DynamicSemanticCircuitStageError(
                "selector optional must be boolean."
            )
        object.__setattr__(self, "circuit_kinds", kinds)
        object.__setattr__(self, "parameter_semantics", semantics)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "circuit_kinds": list(self.circuit_kinds),
            "parameter_semantics": list(self.parameter_semantics),
            "optional": self.optional,
        }


@dataclass(frozen=True, slots=True)
class SemanticCircuitWriteHandoff:
    """Explicitly transfer overlapping write access from one prior stage."""

    from_stage_id: str
    selectors: tuple[SemanticCircuitSelector, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "from_stage_id",
            _stable_name(self.from_stage_id, "handoff from_stage_id"),
        )
        selectors = tuple(self.selectors)
        if not selectors or any(
            not isinstance(value, SemanticCircuitSelector) for value in selectors
        ):
            raise DynamicSemanticCircuitStageError(
                "A write handoff needs SemanticCircuitSelector objects."
            )
        if len({value.fingerprint for value in selectors}) != len(selectors):
            raise DynamicSemanticCircuitStageError(
                "Write handoff selectors must be unique."
            )
        object.__setattr__(
            self,
            "selectors",
            tuple(sorted(selectors, key=lambda value: value.fingerprint)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "from_stage_id": self.from_stage_id,
            "selectors": [value.as_dict() for value in self.selectors],
        }


@dataclass(frozen=True, slots=True)
class SemanticCircuitStageContract:
    """Path-free contract for one mandatory semantic compiler invocation."""

    stage_id: str
    stage_role: CompileStage
    dependencies: tuple[str, ...]
    required_selectors: tuple[SemanticCircuitSelector, ...]
    mutation_selectors: tuple[SemanticCircuitSelector, ...]
    receipt_partitions: tuple[tuple[str, SemanticCircuitSelector], ...]
    write_handoffs: tuple[SemanticCircuitWriteHandoff, ...] = ()
    requires_effectful_write: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stage_id",
            _stable_name(self.stage_id, "stage_id"),
        )
        if not isinstance(self.stage_role, CompileStage):
            raise DynamicSemanticCircuitStageError(
                "stage_role must be CompileStage metadata."
            )
        dependencies = _unique_names(self.dependencies, "stage dependency")
        if self.stage_id in dependencies:
            raise DynamicSemanticCircuitStageError(
                f"Stage {self.stage_id!r} cannot depend on itself."
            )
        object.__setattr__(self, "dependencies", dependencies)
        for name in ("required_selectors", "mutation_selectors"):
            selectors = tuple(getattr(self, name))
            if any(
                not isinstance(value, SemanticCircuitSelector)
                for value in selectors
            ):
                raise DynamicSemanticCircuitStageError(
                    f"{name} must contain SemanticCircuitSelector objects."
                )
            if len({value.fingerprint for value in selectors}) != len(selectors):
                raise DynamicSemanticCircuitStageError(
                    f"{name} must not repeat a selector."
                )
            object.__setattr__(
                self,
                name,
                tuple(sorted(selectors, key=lambda value: value.fingerprint)),
            )
        if not self.mutation_selectors:
            raise DynamicSemanticCircuitStageError(
                "Every stage must declare at least one write-access selector."
            )
        partitions: list[tuple[str, SemanticCircuitSelector]] = []
        for value in self.receipt_partitions:
            if not isinstance(value, tuple) or len(value) != 2:
                raise DynamicSemanticCircuitStageError(
                    "receipt_partitions must contain (partition_id, selector) "
                    "pairs."
                )
            partition_id, selector = value
            if not isinstance(selector, SemanticCircuitSelector):
                raise DynamicSemanticCircuitStageError(
                    "A receipt partition selector must be a "
                    "SemanticCircuitSelector."
                )
            partitions.append(
                (_stable_name(partition_id, "receipt partition_id"), selector)
            )
        if len({name for name, _selector in partitions}) != len(partitions):
            raise DynamicSemanticCircuitStageError(
                "Receipt partition identifiers must be unique within a stage."
            )
        object.__setattr__(
            self,
            "receipt_partitions",
            tuple(sorted(partitions, key=lambda value: value[0])),
        )
        handoffs = tuple(self.write_handoffs)
        if any(
            not isinstance(value, SemanticCircuitWriteHandoff)
            for value in handoffs
        ):
            raise DynamicSemanticCircuitStageError(
                "write_handoffs must contain SemanticCircuitWriteHandoff objects."
            )
        if len({value.from_stage_id for value in handoffs}) != len(handoffs):
            raise DynamicSemanticCircuitStageError(
                "A stage can declare only one handoff per prior stage."
            )
        for handoff in handoffs:
            if handoff.from_stage_id not in dependencies:
                raise DynamicSemanticCircuitStageError(
                    f"Stage {self.stage_id!r} must directly depend on handoff "
                    f"source {handoff.from_stage_id!r}."
                )
        object.__setattr__(
            self,
            "write_handoffs",
            tuple(sorted(handoffs, key=lambda value: value.from_stage_id)),
        )
        if not isinstance(self.requires_effectful_write, bool):
            raise DynamicSemanticCircuitStageError(
                "requires_effectful_write must be boolean."
            )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "semantic_circuit_stage_contract",
            "stage_id": self.stage_id,
            "stage_role": self.stage_role.value,
            "dependencies": list(self.dependencies),
            "required_selectors": [
                value.as_dict() for value in self.required_selectors
            ],
            "mutation_selectors": [
                value.as_dict() for value in self.mutation_selectors
            ],
            "receipt_partitions": {
                name: selector.as_dict()
                for name, selector in self.receipt_partitions
            },
            "write_handoffs": [value.as_dict() for value in self.write_handoffs],
            "requires_effectful_write": self.requires_effectful_write,
        }


@dataclass(frozen=True, slots=True)
class SemanticCircuitStageReport:
    """Compiler-supplied payload; observed writes are deliberately absent."""

    partition_receipts: tuple[tuple[str, Mapping[str, Any]], ...]
    operation_audit: Mapping[str, Any]
    stage_receipt: Mapping[str, Any] = field(default_factory=dict)
    report_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        partitions: list[tuple[str, Mapping[str, Any]]] = []
        for value in self.partition_receipts:
            if not isinstance(value, tuple) or len(value) != 2:
                raise DynamicSemanticCircuitStageError(
                    "partition_receipts must contain (partition_id, receipt) pairs."
                )
            name, receipt = value
            partitions.append(
                (
                    _stable_name(name, "stage report partition_id"),
                    _canonical_json_object(receipt, "partition receipt"),
                )
            )
        if len({name for name, _receipt in partitions}) != len(partitions):
            raise DynamicSemanticCircuitStageError(
                "Stage report partition identifiers must be unique."
            )
        object.__setattr__(
            self,
            "partition_receipts",
            tuple(sorted(partitions, key=lambda value: value[0])),
        )
        object.__setattr__(
            self,
            "operation_audit",
            _canonical_json_object(self.operation_audit, "operation_audit"),
        )
        object.__setattr__(
            self,
            "stage_receipt",
            _canonical_json_object(self.stage_receipt, "stage_receipt"),
        )
        object.__setattr__(
            self,
            "report_fingerprint",
            canonical_sha256(self.as_dict()),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "partition_receipts": {
                name: _thaw_json(receipt)
                for name, receipt in self.partition_receipts
            },
            "operation_audit": _thaw_json(self.operation_audit),
            "stage_receipt": _thaw_json(self.stage_receipt),
        }


class SemanticCircuitStageCompiler(Protocol):
    """One explicitly identified family/modality semantic compiler."""

    semantic_stage_resource_identity: str

    def compile_stage(
        self,
        request: SemanticCircuitStageRequest,
        shared_workspace: Any,
        prior_stage_results: Mapping[str, SemanticCircuitStageResult],
    ) -> SemanticCircuitStageReport: ...


@dataclass(frozen=True, slots=True)
class SemanticCircuitLiveParameterTopology:
    """Pointer-free canonical record of one authenticated live parameter path."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    storage_group: str
    object_group: int
    stride: tuple[int, ...]
    storage_offset: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "storage_group": self.storage_group,
            "object_group": self.object_group,
            "stride": list(self.stride),
            "storage_offset": self.storage_offset,
        }


@dataclass(frozen=True, slots=True)
class SemanticCircuitResolvedPartition:
    partition_id: str
    parameter_paths: tuple[str, ...]
    storage_groups: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "partition_id",
            _stable_name(self.partition_id, "resolved partition_id"),
        )
        object.__setattr__(
            self,
            "parameter_paths",
            _unique_names(
                self.parameter_paths,
                "resolved partition parameter path",
            ),
        )
        object.__setattr__(
            self,
            "storage_groups",
            _unique_names(
                self.storage_groups,
                "resolved partition storage_group",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "partition_id": self.partition_id,
            "parameter_paths": list(self.parameter_paths),
            "storage_groups": list(self.storage_groups),
        }


@dataclass(frozen=True, slots=True)
class SemanticCircuitResolvedHandoff:
    from_stage_id: str
    parameter_paths: tuple[str, ...]
    storage_groups: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "from_stage_id",
            _stable_name(self.from_stage_id, "resolved handoff from_stage_id"),
        )
        object.__setattr__(
            self,
            "parameter_paths",
            _unique_names(
                self.parameter_paths,
                "resolved handoff parameter path",
            ),
        )
        object.__setattr__(
            self,
            "storage_groups",
            _unique_names(
                self.storage_groups,
                "resolved handoff storage_group",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "from_stage_id": self.from_stage_id,
            "parameter_paths": list(self.parameter_paths),
            "storage_groups": list(self.storage_groups),
        }


@dataclass(frozen=True, slots=True)
class SemanticCircuitResolvedStage:
    """Graph-derived stage with write, handoff, and final-owner surfaces."""

    contract: SemanticCircuitStageContract
    compiler_identity: BehaviorCallableIdentity
    compiler_resource_identity: str
    required_node_ids: tuple[str, ...]
    required_parameter_paths: tuple[str, ...]
    required_storage_groups: tuple[str, ...]
    write_node_ids: tuple[str, ...]
    write_parameter_paths: tuple[str, ...]
    write_storage_groups: tuple[str, ...]
    final_partitions: tuple[SemanticCircuitResolvedPartition, ...]
    received_handoffs: tuple[SemanticCircuitResolvedHandoff, ...]
    stage_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "compiler_resource_identity",
            _digest(
                self.compiler_resource_identity,
                "compiler_resource_identity",
            ),
        )
        for name in (
            "required_node_ids",
            "required_parameter_paths",
            "required_storage_groups",
            "write_node_ids",
            "write_parameter_paths",
            "write_storage_groups",
        ):
            object.__setattr__(
                self,
                name,
                _unique_names(getattr(self, name), name),
            )
        partitions = tuple(self.final_partitions)
        handoffs = tuple(self.received_handoffs)
        if any(
            not isinstance(value, SemanticCircuitResolvedPartition)
            for value in partitions
        ) or any(
            not isinstance(value, SemanticCircuitResolvedHandoff)
            for value in handoffs
        ):
            raise DynamicSemanticCircuitStageError(
                "Resolved stage partitions or handoffs have the wrong type."
            )
        object.__setattr__(
            self,
            "final_partitions",
            tuple(sorted(partitions, key=lambda value: value.partition_id)),
        )
        object.__setattr__(
            self,
            "received_handoffs",
            tuple(sorted(handoffs, key=lambda value: value.from_stage_id)),
        )
        object.__setattr__(
            self,
            "stage_fingerprint",
            canonical_sha256(self.as_dict()),
        )

    @property
    def stage_id(self) -> str:
        return self.contract.stage_id

    @property
    def stage_role(self) -> CompileStage:
        return self.contract.stage_role

    @property
    def dependencies(self) -> tuple[str, ...]:
        return self.contract.dependencies

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract.as_dict(),
            "contract_fingerprint": self.contract.fingerprint,
            "compiler_identity": self.compiler_identity.as_dict(),
            "compiler_identity_fingerprint": self.compiler_identity.fingerprint,
            "compiler_resource_identity": self.compiler_resource_identity,
            "required_node_ids": list(self.required_node_ids),
            "required_parameter_paths": list(self.required_parameter_paths),
            "required_storage_groups": list(self.required_storage_groups),
            "write_node_ids": list(self.write_node_ids),
            "write_parameter_paths": list(self.write_parameter_paths),
            "write_storage_groups": list(self.write_storage_groups),
            "final_partitions": [
                value.as_dict() for value in self.final_partitions
            ],
            "received_handoffs": [
                value.as_dict() for value in self.received_handoffs
            ],
        }


@dataclass(frozen=True, slots=True)
class SemanticCircuitStageManifest:
    """Authenticated live-topology and complete stage-DAG preflight."""

    child_graph_fingerprint: str
    child_structural_topology_fingerprint: str
    registry_fingerprint: str
    semantic_classifier_identity: BehaviorCallableIdentity
    semantic_classifier_resource_identity: str
    live_parameter_topology: tuple[SemanticCircuitLiveParameterTopology, ...]
    learned_parameter_paths: tuple[str, ...]
    learned_storage_groups: tuple[str, ...]
    initial_state_fingerprint: str
    stages: tuple[SemanticCircuitResolvedStage, ...]
    final_storage_owners: tuple[tuple[str, str, str], ...]
    manifest_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "child_graph_fingerprint",
            "child_structural_topology_fingerprint",
            "registry_fingerprint",
            "semantic_classifier_resource_identity",
            "initial_state_fingerprint",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "learned_parameter_paths",
            _unique_names(self.learned_parameter_paths, "learned parameter path"),
        )
        object.__setattr__(
            self,
            "learned_storage_groups",
            _unique_names(self.learned_storage_groups, "learned storage_group"),
        )
        topology = tuple(self.live_parameter_topology)
        stages = tuple(self.stages)
        owners = tuple(
            sorted(
                (
                    _stable_name(group, "final owner storage_group"),
                    _stable_name(stage, "final owner stage_id"),
                    _stable_name(partition, "final owner partition_id"),
                )
                for group, stage, partition in self.final_storage_owners
            )
        )
        if len({group for group, _stage, _partition in owners}) != len(owners):
            raise DynamicSemanticCircuitStageError(
                "Final storage ownership must be unique."
            )
        object.__setattr__(self, "live_parameter_topology", topology)
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "final_storage_owners", owners)
        object.__setattr__(
            self,
            "manifest_fingerprint",
            canonical_sha256(self.as_dict()),
        )

    @property
    def ordered_stage_ids(self) -> tuple[str, ...]:
        return tuple(value.stage_id for value in self.stages)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "semantic_circuit_stage_manifest",
            "child_graph_fingerprint": self.child_graph_fingerprint,
            "child_structural_topology_fingerprint": (
                self.child_structural_topology_fingerprint
            ),
            "registry_fingerprint": self.registry_fingerprint,
            "semantic_classifier_identity": (
                self.semantic_classifier_identity.as_dict()
            ),
            "semantic_classifier_resource_identity": (
                self.semantic_classifier_resource_identity
            ),
            "live_parameter_topology": [
                value.as_dict() for value in self.live_parameter_topology
            ],
            "learned_parameter_paths": list(self.learned_parameter_paths),
            "learned_storage_groups": list(self.learned_storage_groups),
            "initial_state_fingerprint": self.initial_state_fingerprint,
            "ordered_stage_ids": list(self.ordered_stage_ids),
            "stages": [value.as_dict() for value in self.stages],
            "final_storage_owners": [
                {
                    "storage_group": group,
                    "stage_id": stage,
                    "partition_id": partition,
                }
                for group, stage, partition in self.final_storage_owners
            ],
            "complete_stage_dag_preflight": True,
            "observed_write_enforcement": True,
            "exact_disjoint_final_storage_ownership": True,
        }


@dataclass(frozen=True, slots=True)
class SemanticCircuitStageRequest:
    """One preflighted stage request over the caller-owned same child."""

    child_model: Any
    child_graph: ModelCircuitGraph
    manifest: SemanticCircuitStageManifest
    stage: SemanticCircuitResolvedStage

    @property
    def stage_id(self) -> str:
        return self.stage.stage_id

    @property
    def stage_role(self) -> CompileStage:
        return self.stage.stage_role

    @property
    def dependencies(self) -> tuple[str, ...]:
        return self.stage.dependencies

    @property
    def required_parameter_paths(self) -> tuple[str, ...]:
        return self.stage.required_parameter_paths

    @property
    def write_parameter_paths(self) -> tuple[str, ...]:
        return self.stage.write_parameter_paths

    @property
    def write_storage_groups(self) -> tuple[str, ...]:
        return self.stage.write_storage_groups

    @property
    def final_partitions(self) -> tuple[SemanticCircuitResolvedPartition, ...]:
        return self.stage.final_partitions

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "semantic_circuit_stage_request",
            "child_graph_fingerprint": self.child_graph.fingerprint,
            "manifest_fingerprint": self.manifest.manifest_fingerprint,
            "stage": self.stage.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class SemanticCircuitObservedWrite:
    """Registry-derived bit-level change for one graph storage group."""

    storage_group: str
    parameter_paths: tuple[str, ...]
    changed_parameter_paths: tuple[str, ...]
    storage_sha256_before: str
    storage_sha256_after: str
    path_sha256_changes: tuple[tuple[str, str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "storage_group",
            _stable_name(self.storage_group, "observed storage_group"),
        )
        for name in ("parameter_paths", "changed_parameter_paths"):
            object.__setattr__(self, name, _unique_names(getattr(self, name), name))
        for name in ("storage_sha256_before", "storage_sha256_after"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        changes = tuple(
            (
                _stable_name(path, "observed changed path"),
                _digest(before, "path_sha256_before"),
                _digest(after, "path_sha256_after"),
            )
            for path, before, after in self.path_sha256_changes
        )
        object.__setattr__(
            self,
            "path_sha256_changes",
            tuple(sorted(changes, key=lambda value: value[0])),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "storage_group": self.storage_group,
            "parameter_paths": list(self.parameter_paths),
            "changed_parameter_paths": list(self.changed_parameter_paths),
            "storage_sha256_before": self.storage_sha256_before,
            "storage_sha256_after": self.storage_sha256_after,
            "path_sha256_changes": [
                {"path": path, "before": before, "after": after}
                for path, before, after in self.path_sha256_changes
            ],
        }


@dataclass(frozen=True, slots=True)
class SemanticCircuitFinalPartitionReceipt:
    """One compiler payload bound to graph-derived final storage ownership."""

    partition_id: str
    parameter_paths: tuple[str, ...]
    storage_groups: tuple[str, ...]
    receipt: Mapping[str, Any]
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "partition_id",
            _stable_name(self.partition_id, "partition_id"),
        )
        object.__setattr__(
            self,
            "parameter_paths",
            _unique_names(self.parameter_paths, "partition parameter path"),
        )
        object.__setattr__(
            self,
            "storage_groups",
            _unique_names(self.storage_groups, "partition storage_group"),
        )
        receipt = _canonical_json_object(self.receipt, "partition receipt")
        object.__setattr__(self, "receipt", receipt)
        object.__setattr__(
            self,
            "receipt_sha256",
            canonical_sha256(_thaw_json(receipt)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "partition_id": self.partition_id,
            "parameter_paths": list(self.parameter_paths),
            "storage_groups": list(self.storage_groups),
            "receipt": _thaw_json(self.receipt),
            "receipt_sha256": self.receipt_sha256,
        }


@dataclass(frozen=True, slots=True)
class SemanticCircuitStageResult:
    """Immutable registry-authenticated outcome of one compiler callback."""

    stage_id: str
    manifest_fingerprint: str
    stage_fingerprint: str
    report_fingerprint: str
    partition_receipts: tuple[SemanticCircuitFinalPartitionReceipt, ...]
    observed_writes: tuple[SemanticCircuitObservedWrite, ...]
    operation_audit: Mapping[str, Any]
    stage_receipt: Mapping[str, Any]
    result_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_id", _stable_name(self.stage_id, "stage_id"))
        for name in (
            "manifest_fingerprint",
            "stage_fingerprint",
            "report_fingerprint",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        partitions = tuple(self.partition_receipts)
        writes = tuple(self.observed_writes)
        object.__setattr__(
            self,
            "partition_receipts",
            tuple(sorted(partitions, key=lambda value: value.partition_id)),
        )
        object.__setattr__(
            self,
            "observed_writes",
            tuple(sorted(writes, key=lambda value: value.storage_group)),
        )
        object.__setattr__(
            self,
            "operation_audit",
            _canonical_json_object(self.operation_audit, "operation_audit"),
        )
        object.__setattr__(
            self,
            "stage_receipt",
            _canonical_json_object(self.stage_receipt, "stage_receipt"),
        )
        object.__setattr__(
            self,
            "result_fingerprint",
            canonical_sha256(self.as_dict()),
        )

    @property
    def observed_changed_parameter_paths(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    path
                    for write in self.observed_writes
                    for path in write.changed_parameter_paths
                }
            )
        )

    @property
    def observed_changed_storage_groups(self) -> tuple[str, ...]:
        return tuple(value.storage_group for value in self.observed_writes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "semantic_circuit_stage_result",
            "stage_id": self.stage_id,
            "manifest_fingerprint": self.manifest_fingerprint,
            "stage_fingerprint": self.stage_fingerprint,
            "report_fingerprint": self.report_fingerprint,
            "partition_receipts": [
                value.as_dict() for value in self.partition_receipts
            ],
            "observed_writes": [value.as_dict() for value in self.observed_writes],
            "observed_changed_parameter_paths": list(
                self.observed_changed_parameter_paths
            ),
            "observed_changed_storage_groups": list(
                self.observed_changed_storage_groups
            ),
            "operation_audit": _thaw_json(self.operation_audit),
            "stage_receipt": _thaw_json(self.stage_receipt),
        }


@dataclass(frozen=True, slots=True)
class SemanticCircuitCompilationResult:
    """Complete ordered stage result retaining its authenticated manifest."""

    manifest: SemanticCircuitStageManifest
    stage_results: tuple[SemanticCircuitStageResult, ...]
    result_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        results = tuple(self.stage_results)
        if tuple(value.stage_id for value in results) != self.manifest.ordered_stage_ids:
            raise DynamicSemanticCircuitStageError(
                "Compilation stage results differ from manifest order."
            )
        if any(
            value.manifest_fingerprint != self.manifest.manifest_fingerprint
            for value in results
        ):
            raise DynamicSemanticCircuitStageError(
                "A stage result is bound to another manifest."
            )
        object.__setattr__(self, "stage_results", results)
        object.__setattr__(
            self,
            "result_fingerprint",
            canonical_sha256(self.as_dict()),
        )

    def stage_result(self, stage_id: str) -> SemanticCircuitStageResult:
        for value in self.stage_results:
            if value.stage_id == stage_id:
                return value
        raise KeyError(stage_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "semantic_circuit_compilation_result",
            "manifest": self.manifest.as_dict(),
            "manifest_fingerprint": self.manifest.manifest_fingerprint,
            "stage_results": [value.as_dict() for value in self.stage_results],
        }


@dataclass(frozen=True, slots=True)
class _RegisteredStage:
    contract: SemanticCircuitStageContract
    compiler: SemanticCircuitStageCompiler
    compiler_identity: BehaviorCallableIdentity
    compiler_resource_identity: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract.as_dict(),
            "contract_fingerprint": self.contract.fingerprint,
            "compiler_identity": self.compiler_identity.as_dict(),
            "compiler_identity_fingerprint": self.compiler_identity.fingerprint,
            "compiler_resource_identity": self.compiler_resource_identity,
        }


@dataclass(frozen=True, slots=True)
class _LiveState:
    values: Mapping[str, Any]
    object_ids: Mapping[str, int]
    storage_identities: Mapping[str, tuple[Any, ...]]
    topology: tuple[SemanticCircuitLiveParameterTopology, ...]
    topology_fingerprint: str
    path_sha256: Mapping[str, str]
    storage_sha256: Mapping[str, str]
    state_fingerprint: str


@dataclass(frozen=True, slots=True)
class _ResolvedDraft:
    registration: _RegisteredStage
    required_node_ids: tuple[str, ...]
    required_parameter_paths: tuple[str, ...]
    required_storage_groups: tuple[str, ...]
    write_node_ids: tuple[str, ...]
    write_parameter_paths: tuple[str, ...]
    write_storage_groups: tuple[str, ...]
    final_partitions: tuple[SemanticCircuitResolvedPartition, ...]
    received_handoffs: tuple[SemanticCircuitResolvedHandoff, ...]


@dataclass(frozen=True, slots=True)
class _ControlPlaneSeal:
    child_graph_fingerprint: str
    manifest_fingerprint: str
    registry_fingerprint: str
    ordered_stage_ids: tuple[str, ...]
    stage_fingerprints: Mapping[str, str]


def _storage_identity(value: Any, *, path: str) -> tuple[Any, ...]:
    try:
        storage = value.untyped_storage()
        pointer = int(storage.data_ptr())
        handle = int(getattr(storage, "_cdata", 0))
        nbytes = int(storage.nbytes())
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise DynamicSemanticCircuitStageError(
            f"Could not authenticate live storage for {path!r}."
        ) from exc
    device = str(getattr(value, "device", ""))
    if pointer:
        return ("storage", device, pointer, nbytes)
    if handle:
        return ("storage_handle", device, handle, nbytes)
    return ("storage_object", device, id(storage), nbytes)


def _raw_storage_sha256(value: Any, *, path: str) -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise DynamicSemanticCircuitStageError(
            "PyTorch is required for observed semantic stage writes."
        ) from exc
    try:
        storage = value.untyped_storage()
        count = int(storage.nbytes())
        raw = torch.empty(0, dtype=torch.uint8, device=value.device).set_(
            storage,
            0,
            (count,),
            (1,),
        )
        payload = raw.detach().to(device="cpu").contiguous().numpy().tobytes()
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise DynamicSemanticCircuitStageError(
            f"Could not hash live storage for {path!r}."
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _path_sha256(value: Any, *, path: str) -> str:
    try:
        contiguous = value.detach().to(device="cpu").contiguous().clone()
        raw_sha256 = _raw_storage_sha256(contiguous, path=path)
        return canonical_sha256(
            {
                "dtype": _normalize_dtype(value.dtype),
                "shape": [int(size) for size in value.shape],
                "raw_sha256": raw_sha256,
            }
        )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise DynamicSemanticCircuitStageError(
            f"Could not hash live parameter path {path!r}."
        ) from exc


class SemanticCircuitStageRegistry:
    """Frozen registry for one storage-authenticated semantic stage DAG.

    There is intentionally no acceptance, publication, skipping, commit, or
    rollback surface.  ``compile`` invokes all preflighted stages exactly once.
    """

    def __init__(
        self,
        *,
        semantic_classifier: SemanticParameterClassifier | None = None,
        semantic_classifier_resource_identity: str | None = None,
        custom_circuit_kinds: Sequence[str] = (),
        custom_parameter_semantics: Sequence[str] = (),
    ) -> None:
        classifier: SemanticParameterClassifier = (
            _DefaultSemanticParameterClassifier()
            if semantic_classifier is None
            else semantic_classifier
        )
        observed_resource = _digest(
            getattr(classifier, "semantic_classifier_resource_identity", None),
            "semantic classifier explicit resource identity",
        )
        declared_resource = (
            observed_resource
            if semantic_classifier_resource_identity is None
            else _digest(
                semantic_classifier_resource_identity,
                "semantic_classifier_resource_identity",
            )
        )
        if declared_resource != observed_resource:
            raise DynamicSemanticCircuitStageError(
                "Semantic classifier explicit resource identity differs from "
                "its declared resource."
            )
        if not callable(classifier):
            raise DynamicSemanticCircuitStageError(
                "semantic_classifier must be callable."
            )
        self._semantic_classifier = classifier
        self._semantic_classifier_resource_identity = declared_resource
        self._semantic_classifier_identity = identify_behavior_callable(
            classifier,
            implementation_id="neural-forge.semantic-circuit-stage.classifier",
            version=1,
        )
        self._known_circuit_kinds = frozenset(
            {value.value for value in CircuitKind}
            | {
                _stable_name(value, "custom circuit kind")
                for value in custom_circuit_kinds
            }
        )
        self._known_parameter_semantics = frozenset(
            _KNOWN_SEMANTICS
            | {
                _stable_name(value, "custom parameter semantic").lower()
                for value in custom_parameter_semantics
            }
        )
        self._registrations: dict[str, _RegisteredStage] = {}
        self._frozen = False
        self._frozen_fingerprint: str | None = None

    def register(
        self,
        contract: SemanticCircuitStageContract,
        compiler: SemanticCircuitStageCompiler,
        *,
        resource_identity: str,
        implementation_id: str | None = None,
        version: int = 1,
    ) -> None:
        if self._frozen:
            raise DynamicSemanticCircuitStageError(
                "Semantic circuit stage registry is frozen."
            )
        if not isinstance(contract, SemanticCircuitStageContract):
            raise DynamicSemanticCircuitStageError(
                "contract must be SemanticCircuitStageContract."
            )
        if contract.stage_id in self._registrations:
            raise DynamicSemanticCircuitStageError(
                f"Stage {contract.stage_id!r} is already registered."
            )
        compile_stage = getattr(compiler, "compile_stage", None)
        if not callable(compile_stage):
            raise DynamicSemanticCircuitStageError(
                "compiler must expose compile_stage(request, shared_workspace, "
                "prior_stage_results)."
            )
        declared_resource = _digest(resource_identity, "compiler resource_identity")
        observed_resource = _digest(
            getattr(compiler, "semantic_stage_resource_identity", None),
            "compiler explicit resource identity",
        )
        if declared_resource != observed_resource:
            raise DynamicSemanticCircuitStageError(
                f"Stage {contract.stage_id!r} compiler resource identity differs "
                "from the explicit registration."
            )
        identity = identify_behavior_callable(
            compile_stage,
            implementation_id=(
                f"neural-forge.semantic-circuit-stage.{contract.stage_id}"
                if implementation_id is None
                else _stable_name(implementation_id, "compiler implementation_id")
            ),
            version=_positive_int(version, "compiler version"),
        )
        self._registrations[contract.stage_id] = _RegisteredStage(
            contract=contract,
            compiler=compiler,
            compiler_identity=identity,
            compiler_resource_identity=declared_resource,
        )

    @property
    def contracts(self) -> tuple[SemanticCircuitStageContract, ...]:
        return tuple(
            self._registrations[key].contract
            for key in sorted(self._registrations)
        )

    def _registry_body(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "semantic_classifier_identity": (
                self._semantic_classifier_identity.as_dict()
            ),
            "semantic_classifier_resource_identity": (
                self._semantic_classifier_resource_identity
            ),
            "known_circuit_kinds": sorted(self._known_circuit_kinds),
            "known_parameter_semantics": sorted(self._known_parameter_semantics),
            "registrations": [
                self._registrations[key].as_dict()
                for key in sorted(self._registrations)
            ],
        }

    @property
    def fingerprint(self) -> str:
        observed = canonical_sha256(self._registry_body())
        if self._frozen_fingerprint is not None and observed != self._frozen_fingerprint:
            raise DynamicSemanticCircuitStageError(
                "Frozen semantic stage registry identity changed."
            )
        return observed

    def _verify_classifier(self) -> None:
        observed_resource = _digest(
            getattr(
                self._semantic_classifier,
                "semantic_classifier_resource_identity",
                None,
            ),
            "semantic classifier explicit resource identity",
        )
        if observed_resource != self._semantic_classifier_resource_identity:
            raise DynamicSemanticCircuitStageError(
                "Semantic classifier explicit resource identity changed."
            )
        identity = self._semantic_classifier_identity
        try:
            identify_behavior_callable(
                self._semantic_classifier,
                implementation_id=identity.implementation_id,
                version=identity.version,
                code_fingerprint=identity.code_fingerprint,
                resource_fingerprint=identity.resource_fingerprint,
            )
        except Exception as exc:
            raise DynamicSemanticCircuitStageError(
                "Semantic classifier callable identity changed."
            ) from exc

    @staticmethod
    def _verify_registration(registration: _RegisteredStage) -> None:
        observed_resource = _digest(
            getattr(
                registration.compiler,
                "semantic_stage_resource_identity",
                None,
            ),
            f"stage {registration.contract.stage_id} explicit resource identity",
        )
        if observed_resource != registration.compiler_resource_identity:
            raise DynamicSemanticCircuitStageError(
                f"Stage {registration.contract.stage_id!r} explicit compiler "
                "resource identity changed."
            )
        identity = registration.compiler_identity
        try:
            identify_behavior_callable(
                registration.compiler.compile_stage,
                implementation_id=identity.implementation_id,
                version=identity.version,
                code_fingerprint=identity.code_fingerprint,
                resource_fingerprint=identity.resource_fingerprint,
            )
        except Exception as exc:
            raise DynamicSemanticCircuitStageError(
                f"Stage {registration.contract.stage_id!r} compiler callable "
                "identity changed."
            ) from exc

    def _freeze(self) -> None:
        if not self._registrations:
            raise DynamicSemanticCircuitStageError(
                "Semantic circuit stage registry is empty."
            )
        self._verify_classifier()
        for registration in self._registrations.values():
            self._verify_registration(registration)
        observed = canonical_sha256(self._registry_body())
        if self._frozen:
            if observed != self._frozen_fingerprint:
                raise DynamicSemanticCircuitStageError(
                    "Frozen semantic stage registry identity changed."
                )
            return
        self._frozen = True
        self._frozen_fingerprint = observed

    @staticmethod
    def _topological_order(
        registrations: Mapping[str, _RegisteredStage],
    ) -> tuple[str, ...]:
        known = set(registrations)
        for stage_id, registration in registrations.items():
            unknown = set(registration.contract.dependencies) - known
            if unknown:
                raise DynamicSemanticCircuitStageError(
                    f"Stage {stage_id!r} has unknown dependencies "
                    f"{sorted(unknown)}."
                )
        indegree = {
            stage_id: len(registration.contract.dependencies)
            for stage_id, registration in registrations.items()
        }
        dependents = {stage_id: set() for stage_id in registrations}
        for stage_id, registration in registrations.items():
            for dependency in registration.contract.dependencies:
                dependents[dependency].add(stage_id)
        ready = sorted(
            stage_id for stage_id, degree in indegree.items() if degree == 0
        )
        ordered: list[str] = []
        while ready:
            stage_id = ready.pop(0)
            ordered.append(stage_id)
            for dependent in sorted(dependents[stage_id]):
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    ready.append(dependent)
                    ready.sort()
        if len(ordered) != len(registrations):
            cycle = sorted(
                stage_id for stage_id, degree in indegree.items() if degree > 0
            )
            raise DynamicSemanticCircuitStageError(
                f"Semantic stage dependencies contain a cycle involving {cycle}."
            )
        return tuple(ordered)

    @staticmethod
    def _graph_indexes(
        graph: ModelCircuitGraph,
    ) -> tuple[
        dict[str, ParameterSpec],
        dict[str, str],
        dict[str, tuple[str, ...]],
        dict[str, str],
    ]:
        specs: dict[str, ParameterSpec] = {}
        path_nodes: dict[str, str] = {}
        storage_paths: dict[str, list[str]] = {}
        path_storage: dict[str, str] = {}
        for node in graph.nodes:
            for parameter in node.parameters:
                specs[parameter.path] = parameter
                path_nodes[parameter.path] = node.node_id
                storage_paths.setdefault(parameter.storage_group, []).append(
                    parameter.path
                )
                path_storage[parameter.path] = parameter.storage_group
        return (
            specs,
            path_nodes,
            {
                group: tuple(sorted(paths))
                for group, paths in storage_paths.items()
            },
            path_storage,
        )

    @staticmethod
    def _named_parameters(child_model: Any) -> tuple[tuple[str, Any], ...]:
        method = getattr(child_model, "named_parameters", None)
        if not callable(method):
            raise DynamicSemanticCircuitStageError(
                "The semantic stage child must expose named_parameters()."
            )
        try:
            values = tuple(method(recurse=True, remove_duplicate=False))
        except TypeError:  # pragma: no cover - old Torch compatibility
            values = tuple(method(recurse=True))
        names = tuple(name for name, _value in values)
        if any(not isinstance(name, str) or not name for name in names):
            raise DynamicSemanticCircuitStageError(
                "Live child parameter paths must be nonempty strings."
            )
        if len(set(names)) != len(names):
            raise DynamicSemanticCircuitStageError(
                "Live child parameter paths must be unique."
            )
        return tuple(sorted(values, key=lambda value: value[0]))

    def _capture_live_state(
        self,
        child_model: Any,
        graph: ModelCircuitGraph,
        *,
        expected: _LiveState | None = None,
    ) -> _LiveState:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency boundary
            raise DynamicSemanticCircuitStageError(
                "PyTorch is required for semantic stage topology authentication."
            ) from exc
        specs, _path_nodes, storage_paths, path_storage = self._graph_indexes(graph)
        values = dict(self._named_parameters(child_model))
        if set(values) != set(specs):
            raise DynamicSemanticCircuitStageError(
                "Live child parameter paths differ from ModelCircuitGraph: "
                f"missing={sorted(set(specs) - set(values))}, "
                f"unknown={sorted(set(values) - set(specs))}."
            )
        object_ids: dict[str, int] = {}
        storage_identities: dict[str, tuple[Any, ...]] = {}
        graph_group_actual: dict[str, set[tuple[Any, ...]]] = {}
        actual_group_graph: dict[tuple[Any, ...], set[str]] = {}
        for path, value in sorted(values.items()):
            if not isinstance(value, torch.Tensor):
                raise DynamicSemanticCircuitStageError(
                    f"Live parameter {path!r} is not a tensor."
                )
            spec = specs[path]
            shape = tuple(int(size) for size in value.shape)
            if shape != spec.shape:
                raise DynamicSemanticCircuitStageError(
                    f"Live parameter {path!r} shape {shape} differs from graph "
                    f"shape {spec.shape}."
                )
            dtype = _normalize_dtype(value.dtype)
            if dtype != _normalize_dtype(spec.dtype):
                raise DynamicSemanticCircuitStageError(
                    f"Live parameter {path!r} dtype {dtype!r} differs from graph "
                    f"dtype {_normalize_dtype(spec.dtype)!r}."
                )
            identity = _storage_identity(value, path=path)
            object_ids[path] = id(value)
            storage_identities[path] = identity
            group = path_storage[path]
            graph_group_actual.setdefault(group, set()).add(identity)
            actual_group_graph.setdefault(identity, set()).add(group)
        split = {
            group: identities
            for group, identities in graph_group_actual.items()
            if len(identities) != 1
        }
        merged = {
            identity: groups
            for identity, groups in actual_group_graph.items()
            if len(groups) != 1
        }
        if split or merged:
            raise DynamicSemanticCircuitStageError(
                "Live parameter tie/storage topology differs from graph "
                f"storage_group declarations: split={sorted(split)}, "
                f"merged={sorted(tuple(sorted(value)) for value in merged.values())}."
            )
        object_group_ids: dict[int, int] = {}
        topology: list[SemanticCircuitLiveParameterTopology] = []
        for path, value in sorted(values.items()):
            object_group = object_group_ids.setdefault(
                id(value),
                len(object_group_ids),
            )
            topology.append(
                SemanticCircuitLiveParameterTopology(
                    path=path,
                    shape=tuple(int(size) for size in value.shape),
                    dtype=_normalize_dtype(value.dtype),
                    storage_group=path_storage[path],
                    object_group=object_group,
                    stride=tuple(int(stride) for stride in value.stride()),
                    storage_offset=int(value.storage_offset()),
                )
            )
        topology_tuple = tuple(topology)
        topology_fingerprint = canonical_sha256(
            [value.as_dict() for value in topology_tuple]
        )
        if expected is not None:
            if topology_fingerprint != expected.topology_fingerprint:
                raise DynamicSemanticCircuitStageError(
                    "Live child parameter topology changed during stage compilation."
                )
            if object_ids != dict(expected.object_ids):
                raise DynamicSemanticCircuitStageError(
                    "Live child Parameter object identity changed during stage "
                    "compilation."
                )
            if storage_identities != dict(expected.storage_identities):
                raise DynamicSemanticCircuitStageError(
                    "Live child storage identity changed during stage compilation."
                )
        path_sha256 = {
            path: _path_sha256(value, path=path)
            for path, value in sorted(values.items())
        }
        storage_sha256 = {
            group: _raw_storage_sha256(values[paths[0]], path=paths[0])
            for group, paths in sorted(storage_paths.items())
        }
        state_fingerprint = canonical_sha256(
            {
                "path_sha256": path_sha256,
                "storage_sha256": storage_sha256,
            }
        )
        return _LiveState(
            values=MappingProxyType(values),
            object_ids=MappingProxyType(object_ids),
            storage_identities=MappingProxyType(storage_identities),
            topology=topology_tuple,
            topology_fingerprint=topology_fingerprint,
            path_sha256=MappingProxyType(path_sha256),
            storage_sha256=MappingProxyType(storage_sha256),
            state_fingerprint=state_fingerprint,
        )

    def _classify_graph(
        self,
        graph: ModelCircuitGraph,
    ) -> dict[str, tuple[str, ...]]:
        self._verify_classifier()
        result: dict[str, tuple[str, ...]] = {}
        call_error: BaseException | None = None
        try:
            for node in graph.nodes:
                if node.kind not in self._known_circuit_kinds:
                    raise DynamicSemanticCircuitStageError(
                        f"Graph circuit kind {node.kind!r} is not authenticated by "
                        "the semantic registry vocabulary."
                    )
                for parameter in node.parameters:
                    observed = self._semantic_classifier(node.kind, parameter.role)
                    if isinstance(observed, str):
                        values = (observed,)
                    else:
                        values = tuple(observed)
                    semantics = tuple(
                        sorted(
                            {
                                _stable_name(
                                    value,
                                    "classified parameter semantic",
                                ).lower()
                                for value in values
                            }
                        )
                    )
                    if not semantics:
                        raise DynamicSemanticCircuitStageError(
                            f"Semantic classifier returned no role for "
                            f"{parameter.path!r}."
                        )
                    unknown = set(semantics) - self._known_parameter_semantics
                    if unknown:
                        raise DynamicSemanticCircuitStageError(
                            f"Semantic classifier returned unauthenticated roles "
                            f"{sorted(unknown)} for {parameter.path!r}."
                        )
                    result[parameter.path] = semantics
        except BaseException as exc:
            call_error = exc
        try:
            self._verify_classifier()
        except BaseException as identity_error:
            if call_error is not None:
                raise identity_error from call_error
            raise
        if call_error is not None:
            raise call_error
        return result

    def _validate_selector_vocabulary(
        self,
        selector: SemanticCircuitSelector,
        *,
        stage_id: str,
    ) -> None:
        unknown_kinds = set(selector.circuit_kinds) - self._known_circuit_kinds
        unknown_semantics = (
            set(selector.parameter_semantics) - self._known_parameter_semantics
        )
        if unknown_kinds or unknown_semantics:
            raise DynamicSemanticCircuitStageError(
                f"Stage {stage_id!r} selector uses unauthenticated vocabulary: "
                f"circuit_kinds={sorted(unknown_kinds)}, "
                f"parameter_semantics={sorted(unknown_semantics)}."
            )

    def _resolve_selector(
        self,
        graph: ModelCircuitGraph,
        selector: SemanticCircuitSelector,
        semantics_by_path: Mapping[str, tuple[str, ...]],
        *,
        stage_id: str,
        selector_name: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        self._validate_selector_vocabulary(selector, stage_id=stage_id)
        _specs, path_nodes, storage_paths, path_storage = self._graph_indexes(graph)
        nodes = (
            tuple(
                node
                for kind in selector.circuit_kinds
                for node in graph.nodes_of_kind(kind)
            )
            if selector.circuit_kinds
            else graph.nodes
        )
        selected_paths: set[str] = set()
        selected_semantics = set(selector.parameter_semantics)
        for node in nodes:
            for path in node.parameter_paths:
                if selected_semantics and not selected_semantics.intersection(
                    semantics_by_path[path]
                ):
                    continue
                selected_paths.add(path)
        if not selected_paths:
            if selector.optional:
                return (), (), ()
            raise DynamicSemanticCircuitStageError(
                f"Stage {stage_id!r} {selector_name} selector "
                f"{selector.fingerprint} matches no learned child parameter."
            )
        groups = {path_storage[path] for path in selected_paths}
        closed_paths = {
            path for group in groups for path in storage_paths[group]
        }
        closed_nodes = {path_nodes[path] for path in closed_paths}
        return (
            tuple(sorted(closed_nodes)),
            tuple(sorted(closed_paths)),
            tuple(sorted(groups)),
        )

    def _resolve_contracts(
        self,
        graph: ModelCircuitGraph,
        ordered_stage_ids: tuple[str, ...],
        semantics_by_path: Mapping[str, tuple[str, ...]],
    ) -> tuple[tuple[SemanticCircuitResolvedStage, ...], tuple[tuple[str, str, str], ...]]:
        drafts: dict[str, _ResolvedDraft] = {}
        for stage_id in ordered_stage_ids:
            registration = self._registrations[stage_id]
            contract = registration.contract
            required_nodes: set[str] = set()
            required_paths: set[str] = set()
            required_groups: set[str] = set()
            for index, selector in enumerate(contract.required_selectors):
                nodes, paths, groups = self._resolve_selector(
                    graph,
                    selector,
                    semantics_by_path,
                    stage_id=stage_id,
                    selector_name=f"required[{index}]",
                )
                required_nodes.update(nodes)
                required_paths.update(paths)
                required_groups.update(groups)
            write_nodes: set[str] = set()
            write_paths: set[str] = set()
            write_groups: set[str] = set()
            for index, selector in enumerate(contract.mutation_selectors):
                nodes, paths, groups = self._resolve_selector(
                    graph,
                    selector,
                    semantics_by_path,
                    stage_id=stage_id,
                    selector_name=f"write[{index}]",
                )
                write_nodes.update(nodes)
                write_paths.update(paths)
                write_groups.update(groups)
            if contract.requires_effectful_write and not write_groups:
                raise DynamicSemanticCircuitStageError(
                    f"Effectful stage {stage_id!r} resolves no write storage."
                )
            final_partitions: list[SemanticCircuitResolvedPartition] = []
            locally_owned: dict[str, str] = {}
            for partition_id, selector in contract.receipt_partitions:
                _nodes, paths, groups = self._resolve_selector(
                    graph,
                    selector,
                    semantics_by_path,
                    stage_id=stage_id,
                    selector_name=f"final_receipt[{partition_id}]",
                )
                outside = set(groups) - write_groups
                overlap = set(groups).intersection(locally_owned)
                if outside:
                    raise DynamicSemanticCircuitStageError(
                        f"Stage {stage_id!r} final partition {partition_id!r} "
                        f"owns storage outside its write access: {sorted(outside)}."
                    )
                if overlap:
                    raise DynamicSemanticCircuitStageError(
                        f"Stage {stage_id!r} final receipt partitions overlap on "
                        f"storage groups {sorted(overlap)}."
                    )
                for group in groups:
                    locally_owned[group] = partition_id
                final_partitions.append(
                    SemanticCircuitResolvedPartition(
                        partition_id=partition_id,
                        parameter_paths=paths,
                        storage_groups=groups,
                    )
                )
            handoffs: list[SemanticCircuitResolvedHandoff] = []
            claimed_handoff_groups: dict[str, str] = {}
            for handoff in contract.write_handoffs:
                paths: set[str] = set()
                groups: set[str] = set()
                for index, selector in enumerate(handoff.selectors):
                    _nodes, resolved_paths, resolved_groups = self._resolve_selector(
                        graph,
                        selector,
                        semantics_by_path,
                        stage_id=stage_id,
                        selector_name=(
                            f"handoff[{handoff.from_stage_id}][{index}]"
                        ),
                    )
                    paths.update(resolved_paths)
                    groups.update(resolved_groups)
                overlap = groups.intersection(claimed_handoff_groups)
                if overlap:
                    raise DynamicSemanticCircuitStageError(
                        f"Stage {stage_id!r} handoffs overlap on storage groups "
                        f"{sorted(overlap)}."
                    )
                for group in groups:
                    claimed_handoff_groups[group] = handoff.from_stage_id
                handoffs.append(
                    SemanticCircuitResolvedHandoff(
                        from_stage_id=handoff.from_stage_id,
                        parameter_paths=tuple(paths),
                        storage_groups=tuple(groups),
                    )
                )
            drafts[stage_id] = _ResolvedDraft(
                registration=registration,
                required_node_ids=tuple(required_nodes),
                required_parameter_paths=tuple(required_paths),
                required_storage_groups=tuple(required_groups),
                write_node_ids=tuple(write_nodes),
                write_parameter_paths=tuple(write_paths),
                write_storage_groups=tuple(write_groups),
                final_partitions=tuple(final_partitions),
                received_handoffs=tuple(handoffs),
            )

        final_owners: dict[str, tuple[str, str]] = {}
        for stage_id, draft in drafts.items():
            for partition in draft.final_partitions:
                for group in partition.storage_groups:
                    if group in final_owners:
                        raise DynamicSemanticCircuitStageError(
                            f"Final receipt storage_group {group!r} is owned by "
                            "more than one stage."
                        )
                    final_owners[group] = (stage_id, partition.partition_id)
        _specs, _path_nodes, storage_paths, _path_storage = self._graph_indexes(graph)
        learned_groups = set(storage_paths)
        if set(final_owners) != learned_groups:
            raise DynamicSemanticCircuitStageError(
                "Final receipt partitions do not exactly own every learned "
                f"storage group: missing={sorted(learned_groups - set(final_owners))}, "
                f"unknown={sorted(set(final_owners) - learned_groups)}."
            )

        order_index = {stage_id: index for index, stage_id in enumerate(ordered_stage_ids)}
        writers = {
            group: tuple(
                sorted(
                    (
                        stage_id
                        for stage_id, draft in drafts.items()
                        if group in draft.write_storage_groups
                    ),
                    key=order_index.__getitem__,
                )
            )
            for group in learned_groups
        }
        for group, chain in writers.items():
            if not chain:
                raise DynamicSemanticCircuitStageError(
                    f"Learned storage_group {group!r} has no writer."
                )
            for previous, current in zip(chain, chain[1:], strict=False):
                current_contract = drafts[current].registration.contract
                handoff_sources = {
                    handoff.from_stage_id
                    for handoff in drafts[current].received_handoffs
                    if group in handoff.storage_groups
                }
                if (
                    previous not in current_contract.dependencies
                    or handoff_sources != {previous}
                ):
                    raise DynamicSemanticCircuitStageError(
                        f"Overlapping writers for storage_group {group!r} lack "
                        f"the explicit adjacent handoff {previous!r} -> {current!r}."
                    )
            final_stage, _partition = final_owners[group]
            if final_stage != chain[-1]:
                raise DynamicSemanticCircuitStageError(
                    f"Final owner {final_stage!r} is not the last writer for "
                    f"storage_group {group!r}; last writer is {chain[-1]!r}."
                )
        for stage_id, draft in drafts.items():
            for handoff in draft.received_handoffs:
                source_groups = set(drafts[handoff.from_stage_id].write_storage_groups)
                valid = source_groups.intersection(draft.write_storage_groups)
                extra = set(handoff.storage_groups) - valid
                if extra:
                    raise DynamicSemanticCircuitStageError(
                        f"Stage {stage_id!r} handoff from "
                        f"{handoff.from_stage_id!r} includes non-overlapping "
                        f"storage groups {sorted(extra)}."
                    )
                for group in handoff.storage_groups:
                    chain = writers[group]
                    position = chain.index(stage_id)
                    if position == 0 or chain[position - 1] != handoff.from_stage_id:
                        raise DynamicSemanticCircuitStageError(
                            f"Stage {stage_id!r} declares a non-adjacent handoff "
                            f"for storage_group {group!r}."
                        )

        resolved = tuple(
            SemanticCircuitResolvedStage(
                contract=drafts[stage_id].registration.contract,
                compiler_identity=drafts[stage_id].registration.compiler_identity,
                compiler_resource_identity=(
                    drafts[stage_id].registration.compiler_resource_identity
                ),
                required_node_ids=drafts[stage_id].required_node_ids,
                required_parameter_paths=drafts[stage_id].required_parameter_paths,
                required_storage_groups=drafts[stage_id].required_storage_groups,
                write_node_ids=drafts[stage_id].write_node_ids,
                write_parameter_paths=drafts[stage_id].write_parameter_paths,
                write_storage_groups=drafts[stage_id].write_storage_groups,
                final_partitions=drafts[stage_id].final_partitions,
                received_handoffs=drafts[stage_id].received_handoffs,
            )
            for stage_id in ordered_stage_ids
        )
        owner_rows = tuple(
            (group, stage, partition)
            for group, (stage, partition) in sorted(final_owners.items())
        )
        return resolved, owner_rows

    def _preflight(
        self,
        child_model: Any,
        child_graph: ModelCircuitGraph,
    ) -> tuple[SemanticCircuitStageManifest, _LiveState]:
        if not isinstance(child_graph, ModelCircuitGraph):
            raise DynamicSemanticCircuitStageError(
                "child_graph must be an authenticated ModelCircuitGraph."
            )
        graph_fingerprint = child_graph.fingerprint
        initial_live = self._capture_live_state(child_model, child_graph)
        semantics = self._classify_graph(child_graph)
        ordered = self._topological_order(self._registrations)
        stages, final_owners = self._resolve_contracts(
            child_graph,
            ordered,
            semantics,
        )
        if child_graph.fingerprint != graph_fingerprint:
            raise DynamicSemanticCircuitStageError(
                "Child graph changed during semantic stage preflight."
            )
        self._freeze()
        authenticated_live = self._capture_live_state(
            child_model,
            child_graph,
            expected=initial_live,
        )
        if authenticated_live.state_fingerprint != initial_live.state_fingerprint:
            raise DynamicSemanticCircuitStageError(
                "Semantic classifier, registry, or resource authentication "
                "mutated learned child state during preflight."
            )
        _specs, _path_nodes, storage_paths, _path_storage = self._graph_indexes(
            child_graph
        )
        manifest = SemanticCircuitStageManifest(
            child_graph_fingerprint=graph_fingerprint,
            child_structural_topology_fingerprint=(
                child_graph.structural_topology_fingerprint
            ),
            registry_fingerprint=self.fingerprint,
            semantic_classifier_identity=self._semantic_classifier_identity,
            semantic_classifier_resource_identity=(
                self._semantic_classifier_resource_identity
            ),
            live_parameter_topology=authenticated_live.topology,
            learned_parameter_paths=tuple(sorted(child_graph.parameter_paths)),
            learned_storage_groups=tuple(sorted(storage_paths)),
            initial_state_fingerprint=authenticated_live.state_fingerprint,
            stages=stages,
            final_storage_owners=final_owners,
        )
        return manifest, authenticated_live

    def preflight(
        self,
        child_model: Any,
        child_graph: ModelCircuitGraph,
    ) -> SemanticCircuitStageManifest:
        """Freeze and authenticate the complete live DAG without callbacks."""

        manifest, _live = self._preflight(child_model, child_graph)
        return manifest

    def _capture_control_plane_seal(
        self,
        *,
        child_graph: ModelCircuitGraph,
        manifest: SemanticCircuitStageManifest,
    ) -> _ControlPlaneSeal:
        stage_fingerprints = MappingProxyType(
            {
                stage.stage_id: stage.stage_fingerprint
                for stage in manifest.stages
            }
        )
        seal = _ControlPlaneSeal(
            child_graph_fingerprint=child_graph.fingerprint,
            manifest_fingerprint=manifest.manifest_fingerprint,
            registry_fingerprint=manifest.registry_fingerprint,
            ordered_stage_ids=manifest.ordered_stage_ids,
            stage_fingerprints=stage_fingerprints,
        )
        self._reauthenticate_control_plane(
            child_graph=child_graph,
            manifest=manifest,
            seal=seal,
            stage_results={},
            result_fingerprints={},
            phase="initial control-plane seal",
            verify_resources=False,
        )
        return seal

    def _reauthenticate_control_plane(
        self,
        *,
        child_graph: ModelCircuitGraph,
        manifest: SemanticCircuitStageManifest,
        seal: _ControlPlaneSeal,
        stage_results: Mapping[str, SemanticCircuitStageResult],
        result_fingerprints: Mapping[str, str],
        phase: str,
        verify_resources: bool,
    ) -> None:
        """Recompute every callback-visible control-plane identity."""

        if verify_resources:
            self._freeze()
        try:
            graph_fingerprint = child_graph.fingerprint
            manifest_body_fingerprint = canonical_sha256(manifest.as_dict())
            registry_body_fingerprint = canonical_sha256(self._registry_body())
            observed_stage_ids = tuple(
                stage.stage_id for stage in manifest.stages
            )
        except Exception as exc:
            raise DynamicSemanticCircuitStageError(
                f"{phase}: control-plane body could not be reauthenticated."
            ) from exc

        if (
            graph_fingerprint != seal.child_graph_fingerprint
            or manifest.child_graph_fingerprint
            != seal.child_graph_fingerprint
        ):
            raise DynamicSemanticCircuitStageError(
                f"{phase}: child graph or manifest graph binding changed."
            )
        if type(manifest) is not SemanticCircuitStageManifest or (
            manifest.manifest_fingerprint != seal.manifest_fingerprint
            or manifest_body_fingerprint != seal.manifest_fingerprint
        ):
            raise DynamicSemanticCircuitStageError(
                f"{phase}: semantic stage manifest body/fingerprint changed."
            )
        if (
            not self._frozen
            or self._frozen_fingerprint != seal.registry_fingerprint
            or manifest.registry_fingerprint != seal.registry_fingerprint
            or registry_body_fingerprint != seal.registry_fingerprint
        ):
            raise DynamicSemanticCircuitStageError(
                f"{phase}: frozen semantic stage registry identity changed."
            )
        if (
            observed_stage_ids != seal.ordered_stage_ids
            or set(seal.stage_fingerprints) != set(observed_stage_ids)
        ):
            raise DynamicSemanticCircuitStageError(
                f"{phase}: resolved semantic stage order/identity changed."
            )
        for expected_stage_id, stage in zip(
            seal.ordered_stage_ids,
            manifest.stages,
            strict=True,
        ):
            expected_fingerprint = seal.stage_fingerprints[expected_stage_id]
            try:
                stage_body_fingerprint = canonical_sha256(stage.as_dict())
            except Exception as exc:
                raise DynamicSemanticCircuitStageError(
                    f"{phase}: resolved stage {expected_stage_id!r} body "
                    "could not be reauthenticated."
                ) from exc
            if (
                type(stage) is not SemanticCircuitResolvedStage
                or stage.stage_id != expected_stage_id
                or stage.stage_fingerprint != expected_fingerprint
                or stage_body_fingerprint != expected_fingerprint
            ):
                raise DynamicSemanticCircuitStageError(
                    f"{phase}: resolved stage {expected_stage_id!r} "
                    "body/fingerprint changed."
                )

        if set(stage_results) != set(result_fingerprints):
            raise DynamicSemanticCircuitStageError(
                f"{phase}: prior stage result identity set changed."
            )
        for stage_id in sorted(result_fingerprints):
            result = stage_results[stage_id]
            expected_fingerprint = result_fingerprints[stage_id]
            try:
                result_body_fingerprint = canonical_sha256(result.as_dict())
            except Exception as exc:
                raise DynamicSemanticCircuitStageError(
                    f"{phase}: prior stage result {stage_id!r} body could "
                    "not be reauthenticated."
                ) from exc
            if (
                type(result) is not SemanticCircuitStageResult
                or result.stage_id != stage_id
                or result.manifest_fingerprint != seal.manifest_fingerprint
                or result.stage_fingerprint
                != seal.stage_fingerprints[stage_id]
                or result.result_fingerprint != expected_fingerprint
                or result_body_fingerprint != expected_fingerprint
            ):
                raise DynamicSemanticCircuitStageError(
                    f"{phase}: prior stage result {stage_id!r} "
                    "body/fingerprint changed."
                )

    @staticmethod
    def _canonical_stage_after_boundary(
        *,
        manifest: SemanticCircuitStageManifest,
        seal: _ControlPlaneSeal,
        expected_stage_id: str,
        retained_stages: Sequence[
            tuple[str, SemanticCircuitResolvedStage]
        ],
        phase: str,
    ) -> SemanticCircuitResolvedStage:
        """Resolve the manifest-owned stage and reject divergent aliases."""

        index = seal.ordered_stage_ids.index(expected_stage_id)
        canonical_stage = manifest.stages[index]
        expected_fingerprint = seal.stage_fingerprints[expected_stage_id]
        for label, retained_stage in (
            ("canonical manifest stage", canonical_stage),
            *retained_stages,
        ):
            try:
                body_fingerprint = canonical_sha256(retained_stage.as_dict())
            except Exception as exc:
                raise DynamicSemanticCircuitStageError(
                    f"{phase}: retained {label} for stage "
                    f"{expected_stage_id!r} could not be reauthenticated."
                ) from exc
            if (
                type(retained_stage) is not SemanticCircuitResolvedStage
                or retained_stage.stage_id != expected_stage_id
                or retained_stage.stage_fingerprint != expected_fingerprint
                or body_fingerprint != expected_fingerprint
            ):
                raise DynamicSemanticCircuitStageError(
                    f"{phase}: retained {label} for stage "
                    f"{expected_stage_id!r} diverged from the authenticated "
                    "manifest stage."
                )
        return canonical_stage

    @staticmethod
    def _observed_writes(
        before: _LiveState,
        after: _LiveState,
        graph: ModelCircuitGraph,
    ) -> tuple[SemanticCircuitObservedWrite, ...]:
        _specs, _path_nodes, storage_paths, _path_storage = (
            SemanticCircuitStageRegistry._graph_indexes(graph)
        )
        writes: list[SemanticCircuitObservedWrite] = []
        for group, paths in sorted(storage_paths.items()):
            storage_before = before.storage_sha256[group]
            storage_after = after.storage_sha256[group]
            path_changes = tuple(
                (
                    path,
                    before.path_sha256[path],
                    after.path_sha256[path],
                )
                for path in paths
                if before.path_sha256[path] != after.path_sha256[path]
            )
            if storage_before == storage_after:
                if path_changes:
                    raise DynamicSemanticCircuitStageError(
                        f"Path hashes changed without their storage hash for "
                        f"storage_group {group!r}."
                    )
                continue
            writes.append(
                SemanticCircuitObservedWrite(
                    storage_group=group,
                    parameter_paths=paths,
                    changed_parameter_paths=tuple(
                        path for path, _before, _after in path_changes
                    ),
                    storage_sha256_before=storage_before,
                    storage_sha256_after=storage_after,
                    path_sha256_changes=path_changes,
                )
            )
        return tuple(writes)

    @staticmethod
    def _authenticate_report(
        *,
        stage_id: str,
        report: SemanticCircuitStageReport,
    ) -> SemanticCircuitStageReport:
        authenticated_report = SemanticCircuitStageReport(
            partition_receipts=tuple(report.partition_receipts),
            operation_audit=report.operation_audit,
            stage_receipt=report.stage_receipt,
        )
        if authenticated_report.report_fingerprint != report.report_fingerprint:
            raise DynamicSemanticCircuitStageError(
                f"Stage {stage_id!r} report changed after its "
                "authenticated construction."
            )
        return authenticated_report

    @staticmethod
    def _result_from_authenticated_report(
        *,
        manifest: SemanticCircuitStageManifest,
        stage: SemanticCircuitResolvedStage,
        report: SemanticCircuitStageReport,
        observed_writes: tuple[SemanticCircuitObservedWrite, ...],
    ) -> SemanticCircuitStageResult:
        expected_partition_ids = tuple(
            value.partition_id for value in stage.final_partitions
        )
        observed_partition_ids = tuple(
            name for name, _receipt in report.partition_receipts
        )
        if observed_partition_ids != expected_partition_ids:
            raise DynamicSemanticCircuitStageError(
                f"Stage {stage.stage_id!r} report partition IDs differ from its "
                f"declared final partitions: expected={expected_partition_ids}, "
                f"observed={observed_partition_ids}."
            )
        payloads = dict(report.partition_receipts)
        receipts = tuple(
            SemanticCircuitFinalPartitionReceipt(
                partition_id=partition.partition_id,
                parameter_paths=partition.parameter_paths,
                storage_groups=partition.storage_groups,
                receipt=payloads[partition.partition_id],
            )
            for partition in stage.final_partitions
        )
        return SemanticCircuitStageResult(
            stage_id=stage.stage_id,
            manifest_fingerprint=manifest.manifest_fingerprint,
            stage_fingerprint=stage.stage_fingerprint,
            report_fingerprint=report.report_fingerprint,
            partition_receipts=receipts,
            observed_writes=observed_writes,
            operation_audit=report.operation_audit,
            stage_receipt=report.stage_receipt,
        )

    def compile(
        self,
        *,
        child_model: Any,
        child_graph: ModelCircuitGraph,
        shared_workspace: Any,
    ) -> SemanticCircuitCompilationResult:
        """Run every stage once; observe writes; leave rollback to the caller."""

        manifest, expected_topology = self._preflight(child_model, child_graph)
        control_seal = self._capture_control_plane_seal(
            child_graph=child_graph,
            manifest=manifest,
        )
        results: dict[str, SemanticCircuitStageResult] = {}
        result_fingerprints: dict[str, str] = {}
        continuity_fingerprint = manifest.initial_state_fingerprint
        previous_stage_id: str | None = None
        for expected_stage_id, retained_loop_stage in zip(
            control_seal.ordered_stage_ids,
            manifest.stages,
            strict=True,
        ):
            self._reauthenticate_control_plane(
                child_graph=child_graph,
                manifest=manifest,
                seal=control_seal,
                stage_results=results,
                result_fingerprints=result_fingerprints,
                phase=f"before stage {expected_stage_id!r}",
                verify_resources=False,
            )
            stage = self._canonical_stage_after_boundary(
                manifest=manifest,
                seal=control_seal,
                expected_stage_id=expected_stage_id,
                retained_stages=(("loop stage", retained_loop_stage),),
                phase=f"before stage {expected_stage_id!r}",
            )
            registration = self._registrations[expected_stage_id]
            preliminary_before = self._capture_live_state(
                child_model,
                child_graph,
                expected=expected_topology,
            )
            if preliminary_before.state_fingerprint != continuity_fingerprint:
                if previous_stage_id is None:
                    raise DynamicSemanticCircuitStageError(
                        "Stage 1 live state differs from the authenticated "
                        "manifest initial state."
                    )
                raise DynamicSemanticCircuitStageError(
                    "Learned child state continuity changed between adjacent "
                    f"stages {previous_stage_id!r} and {expected_stage_id!r}."
                )
            request = SemanticCircuitStageRequest(
                child_model=child_model,
                child_graph=child_graph,
                manifest=manifest,
                stage=stage,
            )
            prior = MappingProxyType(
                {
                    dependency: results[dependency]
                    for dependency in stage.dependencies
                }
            )
            # State hashing can be material for a large child.  Reauthenticate
            # after that work so the binding check is immediately adjacent to
            # the compiler invocation.  A second capture proves that resource
            # getters and identity verification did not write child state.
            self._freeze()
            self._verify_registration(registration)
            callback_before = self._capture_live_state(
                child_model,
                child_graph,
                expected=expected_topology,
            )
            if (
                callback_before.state_fingerprint
                != preliminary_before.state_fingerprint
            ):
                raise DynamicSemanticCircuitStageError(
                    f"Stage {expected_stage_id!r} identity/resource "
                    "authentication mutated learned child state before its "
                    "callback."
                )
            call_error: BaseException | None = None
            report: Any = None
            try:
                report = registration.compiler.compile_stage(
                    request,
                    shared_workspace,
                    prior,
                )
            except BaseException as exc:
                call_error = exc
            callback_after: _LiveState | None = None
            callback_capture_error: BaseException | None = None
            try:
                # Capture the callback boundary before invoking any post-call
                # resource getter.  Only this interval can be attributed to
                # the stage compiler.
                callback_after = self._capture_live_state(
                    child_model,
                    child_graph,
                    expected=expected_topology,
                )
            except BaseException as exc:
                callback_capture_error = exc
            post_authentication_error: BaseException | None = None
            verified_after: _LiveState | None = None
            try:
                self._freeze()
                self._verify_registration(registration)
                verified_after = self._capture_live_state(
                    child_model,
                    child_graph,
                    expected=expected_topology,
                )
                if (
                    callback_after is not None
                    and verified_after.state_fingerprint
                    != callback_after.state_fingerprint
                ):
                    raise DynamicSemanticCircuitStageError(
                        f"Stage {expected_stage_id!r} identity/resource "
                        "authentication mutated learned child state after its "
                        "callback."
                    )
            except BaseException as exc:
                post_authentication_error = exc
            callback_stage: SemanticCircuitResolvedStage | None = None
            post_callback_control_error: BaseException | None = None
            try:
                self._reauthenticate_control_plane(
                    child_graph=child_graph,
                    manifest=manifest,
                    seal=control_seal,
                    stage_results=results,
                    result_fingerprints=result_fingerprints,
                    phase=f"after callback {expected_stage_id!r}",
                    verify_resources=False,
                )
                callback_stage = self._canonical_stage_after_boundary(
                    manifest=manifest,
                    seal=control_seal,
                    expected_stage_id=expected_stage_id,
                    retained_stages=(
                        ("pre-callback local stage", stage),
                        ("callback request stage", request.stage),
                    ),
                    phase=f"after callback {expected_stage_id!r}",
                )
            except BaseException as exc:
                post_callback_control_error = exc
            if post_callback_control_error is not None:
                cause = (
                    post_authentication_error
                    or callback_capture_error
                    or call_error
                )
                if cause is not None:
                    raise post_callback_control_error from cause
                raise post_callback_control_error
            if post_authentication_error is not None:
                cause = callback_capture_error or call_error
                if cause is not None:
                    raise post_authentication_error from cause
                raise post_authentication_error
            if callback_capture_error is not None:
                if call_error is not None:
                    raise callback_capture_error from call_error
                raise callback_capture_error
            if call_error is not None:
                raise call_error
            if callback_after is None or verified_after is None:  # pragma: no cover
                raise DynamicSemanticCircuitStageError(
                    "Stage callback state sandwich is incomplete."
                )
            if callback_stage is None:  # pragma: no cover - defensive narrowing
                raise DynamicSemanticCircuitStageError(
                    "Post-callback canonical stage resolution is incomplete."
                )
            if type(report) is not SemanticCircuitStageReport:
                raise DynamicSemanticCircuitStageError(
                    f"Stage {expected_stage_id!r} must return an exact "
                    "SemanticCircuitStageReport."
                )
            observed = self._observed_writes(
                callback_before,
                callback_after,
                child_graph,
            )
            self._reauthenticate_control_plane(
                child_graph=child_graph,
                manifest=manifest,
                seal=control_seal,
                stage_results=results,
                result_fingerprints=result_fingerprints,
                phase=f"before write authorization {expected_stage_id!r}",
                verify_resources=False,
            )
            authorization_stage = self._canonical_stage_after_boundary(
                manifest=manifest,
                seal=control_seal,
                expected_stage_id=expected_stage_id,
                retained_stages=(
                    ("pre-callback local stage", stage),
                    ("callback request stage", request.stage),
                    ("post-callback canonical stage", callback_stage),
                ),
                phase=f"before write authorization {expected_stage_id!r}",
            )
            changed_groups = {value.storage_group for value in observed}
            changed_paths = {
                path
                for value in observed
                for path in value.changed_parameter_paths
            }
            unauthorized_groups = changed_groups - set(
                authorization_stage.write_storage_groups
            )
            unauthorized_paths = changed_paths - set(
                authorization_stage.write_parameter_paths
            )
            if unauthorized_groups or unauthorized_paths:
                raise DynamicSemanticCircuitStageError(
                    f"Stage {expected_stage_id!r} performed an unauthorized learned "
                    f"write: storage_groups={sorted(unauthorized_groups)}, "
                    f"parameter_paths={sorted(unauthorized_paths)}."
                )
            if (
                authorization_stage.contract.requires_effectful_write
                and not changed_groups
            ):
                raise DynamicSemanticCircuitStageError(
                    f"Effectful stage {expected_stage_id!r} left every declared "
                    "write storage unchanged."
                )
            authenticated_report: SemanticCircuitStageReport | None = None
            result_error: BaseException | None = None
            try:
                authenticated_report = self._authenticate_report(
                    stage_id=expected_stage_id,
                    report=report,
                )
            except BaseException as exc:
                result_error = exc

            report_stage: SemanticCircuitResolvedStage | None = None
            post_materialization_control_error: BaseException | None = None
            try:
                self._reauthenticate_control_plane(
                    child_graph=child_graph,
                    manifest=manifest,
                    seal=control_seal,
                    stage_results=results,
                    result_fingerprints=result_fingerprints,
                    phase=(
                        f"after report materialization {expected_stage_id!r}"
                    ),
                    verify_resources=True,
                )
                report_stage = self._canonical_stage_after_boundary(
                    manifest=manifest,
                    seal=control_seal,
                    expected_stage_id=expected_stage_id,
                    retained_stages=(
                        ("pre-callback local stage", stage),
                        ("callback request stage", request.stage),
                        ("post-callback canonical stage", callback_stage),
                        ("authorization stage", authorization_stage),
                    ),
                    phase=(
                        f"after report materialization {expected_stage_id!r}"
                    ),
                )
            except BaseException as exc:
                post_materialization_control_error = exc

            stage_result: SemanticCircuitStageResult | None = None
            candidate_results: dict[str, SemanticCircuitStageResult] | None = None
            candidate_result_fingerprints: dict[str, str] | None = None
            if (
                result_error is None
                and post_materialization_control_error is None
                and authenticated_report is not None
                and report_stage is not None
            ):
                try:
                    stage_result = self._result_from_authenticated_report(
                        manifest=manifest,
                        stage=report_stage,
                        report=authenticated_report,
                        observed_writes=observed,
                    )
                    stage_result_body_fingerprint = canonical_sha256(
                        stage_result.as_dict()
                    )
                    if (
                        stage_result.result_fingerprint
                        != stage_result_body_fingerprint
                    ):
                        raise DynamicSemanticCircuitStageError(
                            f"Stage {expected_stage_id!r} result "
                            "body/fingerprint differ immediately after "
                            "construction."
                        )
                    candidate_results = dict(results)
                    candidate_results[expected_stage_id] = stage_result
                    candidate_result_fingerprints = dict(result_fingerprints)
                    candidate_result_fingerprints[expected_stage_id] = (
                        stage_result_body_fingerprint
                    )
                except BaseException as exc:
                    result_error = exc

            post_result_control_error: BaseException | None = None
            if (
                result_error is None
                and post_materialization_control_error is None
                and report_stage is not None
                and candidate_results is not None
                and candidate_result_fingerprints is not None
            ):
                try:
                    self._reauthenticate_control_plane(
                        child_graph=child_graph,
                        manifest=manifest,
                        seal=control_seal,
                        stage_results=candidate_results,
                        result_fingerprints=candidate_result_fingerprints,
                        phase=f"after result construction {expected_stage_id!r}",
                        verify_resources=False,
                    )
                    self._canonical_stage_after_boundary(
                        manifest=manifest,
                        seal=control_seal,
                        expected_stage_id=expected_stage_id,
                        retained_stages=(
                            ("pre-callback local stage", stage),
                            ("callback request stage", request.stage),
                            ("post-callback canonical stage", callback_stage),
                            ("authorization stage", authorization_stage),
                            ("report canonical stage", report_stage),
                        ),
                        phase=f"after result construction {expected_stage_id!r}",
                    )
                except BaseException as exc:
                    post_result_control_error = exc

            final_live: _LiveState | None = None
            final_capture_error: BaseException | None = None
            try:
                # Report authentication and result construction dereference
                # compiler-owned data.  Seal that entire surface with one
                # last live-state capture before accepting the stage result.
                final_live = self._capture_live_state(
                    child_model,
                    child_graph,
                    expected=expected_topology,
                )
            except BaseException as exc:
                final_capture_error = exc
            final_control_error: BaseException | None = None
            if (
                final_capture_error is None
                and final_live is not None
                and result_error is None
                and post_materialization_control_error is None
                and post_result_control_error is None
                and candidate_results is not None
                and candidate_result_fingerprints is not None
            ):
                try:
                    self._reauthenticate_control_plane(
                        child_graph=child_graph,
                        manifest=manifest,
                        seal=control_seal,
                        stage_results=candidate_results,
                        result_fingerprints=candidate_result_fingerprints,
                        phase=f"before accepting stage {expected_stage_id!r}",
                        verify_resources=False,
                    )
                    self._canonical_stage_after_boundary(
                        manifest=manifest,
                        seal=control_seal,
                        expected_stage_id=expected_stage_id,
                        retained_stages=(
                            ("pre-callback local stage", stage),
                            ("callback request stage", request.stage),
                            ("post-callback canonical stage", callback_stage),
                            ("authorization stage", authorization_stage),
                            ("report canonical stage", report_stage),
                        ),
                        phase=f"before accepting stage {expected_stage_id!r}",
                    )
                except BaseException as exc:
                    final_control_error = exc
            if final_capture_error is not None:
                cause = (
                    post_result_control_error
                    or post_materialization_control_error
                    or result_error
                )
                if cause is not None:
                    raise final_capture_error from cause
                raise final_capture_error
            if final_live is None:  # pragma: no cover - defensive narrowing
                raise DynamicSemanticCircuitStageError(
                    "Final stage live-state capture is incomplete."
                )
            if final_live.state_fingerprint != verified_after.state_fingerprint:
                mutation_error = DynamicSemanticCircuitStageError(
                    f"Stage {expected_stage_id!r} report/result materialization "
                    "mutated learned child state after callback authentication."
                )
                cause = (
                    post_result_control_error
                    or post_materialization_control_error
                    or result_error
                )
                if cause is not None:
                    raise mutation_error from cause
                raise mutation_error
            if final_control_error is not None:
                raise final_control_error
            if post_result_control_error is not None:
                raise post_result_control_error
            if post_materialization_control_error is not None:
                if result_error is not None:
                    raise post_materialization_control_error from result_error
                raise post_materialization_control_error
            if result_error is not None:
                raise result_error
            if (
                stage_result is None
                or candidate_results is None
                or candidate_result_fingerprints is None
            ):  # pragma: no cover - defensive narrowing
                raise DynamicSemanticCircuitStageError(
                    "Stage result materialization is incomplete."
                )
            results = candidate_results
            result_fingerprints = candidate_result_fingerprints
            continuity_fingerprint = final_live.state_fingerprint
            previous_stage_id = expected_stage_id
        compilation_result = SemanticCircuitCompilationResult(
            manifest=manifest,
            stage_results=tuple(
                results[stage_id]
                for stage_id in control_seal.ordered_stage_ids
            ),
        )
        if compilation_result.result_fingerprint != canonical_sha256(
            compilation_result.as_dict()
        ):
            raise DynamicSemanticCircuitStageError(
                "Final compilation result body/fingerprint changed."
            )
        self._reauthenticate_control_plane(
            child_graph=child_graph,
            manifest=manifest,
            seal=control_seal,
            stage_results=results,
            result_fingerprints=result_fingerprints,
            phase="before final compilation return",
            verify_resources=False,
        )
        return compilation_result


__all__ = [
    "SCHEMA",
    "DEFAULT_SEMANTIC_CLASSIFIER_RESOURCE_IDENTITY",
    "DynamicSemanticCircuitStageError",
    "SemanticCircuitCompilationResult",
    "SemanticCircuitFinalPartitionReceipt",
    "SemanticCircuitLiveParameterTopology",
    "SemanticCircuitObservedWrite",
    "SemanticCircuitResolvedHandoff",
    "SemanticCircuitResolvedPartition",
    "SemanticCircuitResolvedStage",
    "SemanticCircuitSelector",
    "SemanticCircuitStageCompiler",
    "SemanticCircuitStageContract",
    "SemanticCircuitStageManifest",
    "SemanticCircuitStageRegistry",
    "SemanticCircuitStageReport",
    "SemanticCircuitStageRequest",
    "SemanticCircuitStageResult",
    "SemanticCircuitWriteHandoff",
    "SemanticParameterClassifier",
    "default_semantic_parameter_classifier",
]
