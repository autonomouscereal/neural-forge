"""Sealed accepted-layer restart capsules for the LFM2 closed-loop body fold.

The closed-loop body fold folds every selected layer sequentially inside one
in-memory child transaction.  Before this module there was no mid-body
persistence at all: a process that died at layer 9 of 14 had to re-fold from
layer 0.  A capsule freezes the exact post-accept child state of one layer,
together with the completed-layer inventory, that layer's validation decision
seal, the adapter's coordinate-map/basis identities, and the fold-progress
chain position, so a later process can skip the accepted prefix.

A capsule is restart evidence.  It is never an accepted child, never a
promotable artifact, and never a behavior claim.  Every manifest carries the
fixed non-promotable flag set and every open re-authenticates the canonical
JSON, the manifest seal, the safetensors file bytes, and every tensor's shape,
dtype, and content digest.  Any mismatch raises; nothing warns and continues.

Cross-layer in-memory state
---------------------------

Resuming mid-body is only sound if the state the orchestrator and adapter carry
*between* layers is either stored in the capsule or provably absent.  The audit
of ``ChildClosedLoopCompiler._run_layer`` and ``Lfm2DynamicClosedLoopAdapter``
found the following, and this module is built around it.

Provably not carried across layers:

* ``coordinate_map`` / ``map_sha256`` are constructor-bound and immutable for
  the run.  ``_coordinate_matrix`` re-materializes the dense basis from the map
  on every call and never caches or mutates it, so there is no moving basis
  state.  ``_short_conv_residual_assignment`` and ``_short_conv_assignments``
  are built once in ``__init__`` from that same immutable basis.
* The moving-gauge chart state (``_moving_gauge_row_manifest``,
  ``_moving_gauge_paired_rows``, ``_BodyReclosurePayload``,
  ``_CompactInputProducerReclosureState``) is created and consumed entirely
  inside the model-global body-reclosure stage, which runs *after* the whole
  layer loop.  No moving-gauge value crosses a layer boundary.
* ``_split_transactions[L]`` opens in ``install_producer`` and is popped in
  ``install_consumer`` of the same layer.
* Capture leases, ``_owned_captures``, and ``_active_selected_parent_profile``
  are released at the end of each stage.
* ``_heterogeneous_single_pass_certification_registry`` is a cache keyed by the
  live child identity and the capture selection.  A resumed process rebuilds
  the entries it needs; the only cost is one canonical reference replay for the
  first capture at each new child state.
* ``_producer_payloads[L]``, ``_consumer_payloads[L]`` and
  ``_downstream_payloads[L]`` are written and read within layer L only --
  with one exception, below.

Carried across layers and NOT reconstructible from a capsule:

* ``compile_body_reclosure`` in ``moving_gauge_v3`` mode asserts
  ``set(self._downstream_payloads) == set(range(len(self.child_layers)))``.
  That is not a data dependency on the payload values; it is a precondition
  claim that *this process* compiled and installed every layer of the stack.
  A resumed process that skipped a prefix cannot honestly satisfy it, and
  fabricating placeholder payloads would forge that claim.  Therefore
  :meth:`Lfm2ClosedLoopRestartCapsuleWriter.assert_resume_supported` fails
  closed when a mid-body resume is combined with an enabled body-reclosure
  stage on an adapter whose reclosure mode requires the complete freshly
  installed stack.  Resuming with ``fold_body_reclosure`` disabled, or with the
  adapter in ``disabled`` reclosure mode, is admitted.

Because the capsule stores the exact post-accept child state, no replay of the
skipped layers is required or performed.  The resumed transaction's rollback
point is the capsule state, not the original blank child; that is recorded in
the run report and is the intended semantics of a restart point.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_closed_loop_fold import (
    ClosedLoopAudit,
    ClosedLoopCompilerConfig,
    deterministic_whole_trajectory_split,
)
from neural_forge.hf_lfm2_cpcf_resume_artifact import (
    _atomic_rename_noreplace,
    _file_record,
    _fsync_directory,
    _is_link_or_junction,
    _lexists,
    _load_safetensors_path,
    _normalize_tensors,
    _read_canonical_json_with_raw,
    _real_directory,
    _regular_file,
    _save_safetensors,
    _stat_identity,
    _write_canonical_json,
)


CLOSED_LOOP_RESTART_CAPSULE_SCHEMA = (
    "neural-forge-lfm2-closed-loop-restart-capsule-v1"
)
CLOSED_LOOP_RESTART_CAPSULE_BINDINGS_SCHEMA = (
    "neural-forge-lfm2-closed-loop-restart-capsule-bindings-v1"
)
CLOSED_LOOP_RESTART_CAPSULE_KEY_SCHEMA = (
    "neural-forge-lfm2-closed-loop-restart-capsule-key-v1"
)
CLOSED_LOOP_RESTART_CAPSULE_CHAIN_SCHEMA = (
    "neural-forge-lfm2-closed-loop-restart-capsule-chain-v1"
)
CLOSED_LOOP_RESTART_CAPSULE_CHILD_STATE_SCHEMA = (
    "neural-forge-lfm2-closed-loop-restart-capsule-child-state-v1"
)
CLOSED_LOOP_RESTART_CAPSULE_TOPOLOGY_SCHEMA = (
    "neural-forge-lfm2-closed-loop-restart-capsule-child-topology-v1"
)
CLOSED_LOOP_RESTART_CAPSULE_PROGRESS_SCHEMA = (
    "neural-forge-lfm2-closed-loop-restart-capsule-fold-progress-v1"
)
CLOSED_LOOP_RESTART_CAPSULE_RECORD_SCHEMA = (
    "neural-forge-lfm2-closed-loop-restart-capsule-record-v1"
)
CLOSED_LOOP_RESTART_CAPSULE_WRITER_SCHEMA = (
    "neural-forge-lfm2-closed-loop-restart-capsule-writer-v1"
)
CLOSED_LOOP_RESTART_CAPSULE_RESUME_SCHEMA = (
    "neural-forge-lfm2-closed-loop-restart-capsule-resume-v1"
)
CLOSED_LOOP_RESTART_CAPSULE_TERMINAL_PHASE_SCHEMA = (
    "neural-forge-lfm2-closed-loop-restart-capsule-terminal-phase-v1"
)

CLOSED_LOOP_RESTART_CAPSULE_CHAIN_GENESIS_SHA256 = "0" * 64
DEFAULT_RESTART_CAPSULE_ROTATION_KEEP = 2

_ARTIFACT_KIND = "lfm2-closed-loop-accepted-layer-restart"
_ACCEPTED_LAYER_PHASE = "accepted_body_layer"
_TERMINAL_PHASE = "terminal_post_output_interface"
_CAPSULE_PHASES = frozenset({_ACCEPTED_LAYER_PHASE, _TERMINAL_PHASE})
_STATUS = "quarantined-mid-body-restart-only"
_MANIFEST_NAME = "capsule-manifest.json"
_CHILD_STATE_NAME = "child-state.safetensors"
_CHILD_STATE_FILE_ROLE = "closed-loop-restart-child-state"
_ROOT_INVENTORY = frozenset({_MANIFEST_NAME, _CHILD_STATE_NAME})
_STAGE_PREFIX = ".lfm2-closed-loop-restart-"
_DIRECTORY_PREFIX = "capsule-"

_NONPROMOTABLE_FLAGS = {
    "status": _STATUS,
    "complete_child": False,
    "promotion_authorized": False,
    "accepted_artifact": False,
    "behavior_capability_proven": False,
    "training_performed": False,
    "optimizer_steps": 0,
    "backward_calls": 0,
    "parent_forward_calls": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
    "parent_model_required_for_reload": False,
}

_ZERO_OPERATION_AUDIT = {
    "parent_model_forward_calls": 0,
    "child_model_forward_calls": 0,
    "child_replay_captures": 0,
    "model_generate_calls": 0,
    "training_steps": 0,
    "optimizer_constructions": 0,
    "optimizer_steps": 0,
    "backward_calls": 0,
    "autograd_grad_calls": 0,
    "parameters_pruned": 0,
    "parameters_quantized": 0,
    "network_access_performed": False,
    "pickle_files_written": 0,
}

_ADAPTER_IDENTITY_FIELDS = (
    "adapter_schema",
    "coordinate_map_sha256",
    "child_basis_sha256",
    "child_input_vocabulary_sha256",
    "parent_cache_key",
    "parent_identity_sha256",
    "source_graph_fingerprint",
    "tokenizer_bridge_artifact_sha256",
    "identity_tokenizer_evidence_sha256",
    "parent_residual_operator_manifest_sha256",
    "body_reclosure_mode",
    "body_coordinate_completion_recipe",
    "replay_capture_mode",
    "child_layer_count",
    "selected_layer_indices",
)


class Lfm2ClosedLoopRestartCapsuleError(RuntimeError):
    """A restart capsule failed publication, authentication, or resume."""


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2ClosedLoopRestartCapsuleError(
            f"{name} must be one lowercase SHA-256 digest."
        )
    return value


def _optional_digest(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _digest(value, name)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2ClosedLoopRestartCapsuleError(f"{name} must be nonempty.")
    return value.strip()


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2ClosedLoopRestartCapsuleError(
            f"{name} must be a non-negative integer."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2ClosedLoopRestartCapsuleError(
            f"{name} must be one positive integer."
        )
    return value


def _layer_indices(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise Lfm2ClosedLoopRestartCapsuleError(
            f"{name} must be an ordered sequence of layer indices."
        )
    result = tuple(_nonnegative_int(item, f"{name} entry") for item in value)
    if result != tuple(sorted(set(result))):
        raise Lfm2ClosedLoopRestartCapsuleError(
            f"{name} must be ascending and unique."
        )
    return result


def _sealed(body: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    result = dict(body)
    result[field_name] = canonical_sha256(result)
    return result


def _verify_seal(value: Any, *, field_name: str, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2ClosedLoopRestartCapsuleError(f"{label} must be a mapping.")
    result = dict(value)
    body = dict(result)
    supplied = _digest(body.pop(field_name, None), f"{label} {field_name}")
    if canonical_sha256(body) != supplied:
        raise Lfm2ClosedLoopRestartCapsuleError(f"{label} seal is invalid.")
    return result


def _validate_nonpromotable(value: Mapping[str, Any], *, label: str) -> None:
    if any(
        value.get(name) != expected
        for name, expected in _NONPROMOTABLE_FLAGS.items()
    ):
        raise Lfm2ClosedLoopRestartCapsuleError(
            f"{label} is not fixed non-promotable restart evidence."
        )


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - required LFM2 dependency
        raise Lfm2ClosedLoopRestartCapsuleError(
            "PyTorch is required for closed-loop restart capsules."
        ) from exc
    return torch


def _storage_identity(value: Any) -> tuple[Any, ...]:
    try:
        return (
            str(value.device),
            int(value.untyped_storage().data_ptr()),
            int(value.storage_offset()),
            tuple(int(size) for size in value.shape),
            tuple(int(size) for size in value.stride()),
        )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise Lfm2ClosedLoopRestartCapsuleError(
            "Could not authenticate a child tensor storage identity."
        ) from exc


def _child_state_dict(model: Any) -> dict[str, Any]:
    torch = _torch()
    try:
        state = model.state_dict()
    except Exception as exc:  # pragma: no cover - model specific
        raise Lfm2ClosedLoopRestartCapsuleError(
            "Could not read the child state dictionary."
        ) from exc
    if not isinstance(state, Mapping) or not state:
        raise Lfm2ClosedLoopRestartCapsuleError(
            "The child state dictionary is empty."
        )
    result: dict[str, Any] = {}
    for name, value in state.items():
        if (
            not isinstance(name, str)
            or not name
            or name.strip() != name
            or not isinstance(value, torch.Tensor)
            or value.device.type == "meta"
            or value.layout != torch.strided
        ):
            raise Lfm2ClosedLoopRestartCapsuleError(
                "Child state entries must be named, materialized dense tensors."
            )
        result[name] = value
    return result


def child_state_alias_groups(model: Any) -> tuple[tuple[str, ...], ...]:
    """Group child state entries that share one exact tensor storage view.

    ``safetensors`` refuses shared storages, and an LFM2 child routinely ties
    the input and output token tables.  The capsule therefore stores one
    canonical tensor per storage group and re-proves the group membership of a
    freshly provided child before restoring anything.
    """

    state = _child_state_dict(model)
    groups: dict[tuple[Any, ...], list[str]] = {}
    for name in sorted(state):
        groups.setdefault(_storage_identity(state[name]), []).append(name)
    ordered = sorted(
        (tuple(names) for names in groups.values()),
        key=lambda names: names[0],
    )
    flattened = [name for names in ordered for name in names]
    if sorted(flattened) != sorted(state):
        raise Lfm2ClosedLoopRestartCapsuleError(
            "Child state alias grouping lost or duplicated an entry."
        )
    return tuple(ordered)


def portable_child_state_sha256(model: Any) -> str:
    """Digest child state bytes without any process-local object identity.

    ``Lfm2DynamicClosedLoopAdapter.child_state_fingerprint`` mixes ``id()``
    values of live ``Parameter`` objects, so it cannot be compared across
    processes.  The capsule seals this digest instead and re-proves it after a
    fresh child is loaded.
    """

    state = _child_state_dict(model)
    groups = child_state_alias_groups(model)
    entries = []
    for name in sorted(state):
        value = state[name]
        entries.append(
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": [int(size) for size in value.shape],
                "tensor_sha256": _tensor_content_sha256(value),
            }
        )
    return canonical_sha256(
        {
            "schema": CLOSED_LOOP_RESTART_CAPSULE_CHILD_STATE_SCHEMA,
            "entries": entries,
            "alias_groups": [list(names) for names in groups],
        }
    )


def portable_child_topology_sha256(model: Any) -> str:
    """Digest the child structure without process-local object identity."""

    torch = _torch()
    state = _child_state_dict(model)
    parameters = tuple(model.named_parameters(remove_duplicate=False))
    buffers = tuple(model.named_buffers(remove_duplicate=False))
    body = {
        "schema": CLOSED_LOOP_RESTART_CAPSULE_TOPOLOGY_SCHEMA,
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "state": [
            {
                "name": name,
                "shape": [int(size) for size in state[name].shape],
                "dtype": str(state[name].dtype),
                "layout": str(state[name].layout),
            }
            for name in sorted(state)
        ],
        "parameters": [
            {
                "name": name,
                "shape": [int(size) for size in parameter.shape],
                "dtype": str(parameter.dtype),
                "requires_grad": bool(parameter.requires_grad),
            }
            for name, parameter in sorted(parameters, key=lambda item: item[0])
        ],
        "buffers": [
            {
                "name": name,
                "shape": [int(size) for size in value.shape],
                "dtype": str(value.dtype),
            }
            for name, value in sorted(buffers, key=lambda item: item[0])
            if isinstance(value, torch.Tensor)
        ],
        "alias_groups": [
            list(names) for names in child_state_alias_groups(model)
        ],
    }
    return canonical_sha256(body)


def _tensor_content_sha256(value: Any) -> str:
    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    if tensor.is_conj():
        tensor = tensor.resolve_conj()
    if tensor.is_neg():
        tensor = tensor.resolve_neg()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        ",".join(str(int(size)) for size in tensor.shape).encode("ascii")
    )
    digest.update(b"\0")
    if tensor.numel():
        digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Lfm2ClosedLoopRestartCapsuleBindings:
    """Externally authenticated run identity carried by every capsule."""

    run_id: str
    child_name: str
    child_layer_count: int
    selected_layer_indices: tuple[int, ...]
    adapter_identity_sha256: str
    trajectory_split_sha256: str
    run_schema: str
    bindings_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id"))
        object.__setattr__(
            self, "child_name", _text(self.child_name, "child_name")
        )
        object.__setattr__(
            self, "run_schema", _text(self.run_schema, "run_schema")
        )
        object.__setattr__(
            self,
            "child_layer_count",
            _positive_int(self.child_layer_count, "child_layer_count"),
        )
        selected = _layer_indices(
            self.selected_layer_indices,
            "selected_layer_indices",
        )
        if not selected or selected[-1] >= self.child_layer_count:
            raise Lfm2ClosedLoopRestartCapsuleError(
                "selected_layer_indices must be nonempty and within the child."
            )
        object.__setattr__(self, "selected_layer_indices", selected)
        object.__setattr__(
            self,
            "adapter_identity_sha256",
            _digest(self.adapter_identity_sha256, "adapter_identity_sha256"),
        )
        object.__setattr__(
            self,
            "trajectory_split_sha256",
            _digest(self.trajectory_split_sha256, "trajectory_split_sha256"),
        )
        object.__setattr__(
            self,
            "bindings_sha256",
            canonical_sha256(self.as_dict(include_sha256=False)),
        )

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, Any]:
        result = {
            "schema": CLOSED_LOOP_RESTART_CAPSULE_BINDINGS_SCHEMA,
            "run_id": self.run_id,
            "child_name": self.child_name,
            "child_layer_count": self.child_layer_count,
            "selected_layer_indices": list(self.selected_layer_indices),
            "adapter_identity_sha256": self.adapter_identity_sha256,
            "trajectory_split_sha256": self.trajectory_split_sha256,
            "run_schema": self.run_schema,
        }
        if include_sha256:
            result["bindings_sha256"] = self.bindings_sha256
        return result

    @classmethod
    def from_mapping(
        cls,
        value: Any,
    ) -> "Lfm2ClosedLoopRestartCapsuleBindings":
        if not isinstance(value, Mapping):
            raise Lfm2ClosedLoopRestartCapsuleError(
                "Capsule bindings must be one mapping."
            )
        expected = {
            "schema",
            "run_id",
            "child_name",
            "child_layer_count",
            "selected_layer_indices",
            "adapter_identity_sha256",
            "trajectory_split_sha256",
            "run_schema",
            "bindings_sha256",
        }
        if set(value) != expected or value.get("schema") != (
            CLOSED_LOOP_RESTART_CAPSULE_BINDINGS_SCHEMA
        ):
            raise Lfm2ClosedLoopRestartCapsuleError(
                "Capsule bindings have missing or unknown fields."
            )
        result = cls(
            run_id=value["run_id"],
            child_name=value["child_name"],
            child_layer_count=value["child_layer_count"],
            selected_layer_indices=tuple(value["selected_layer_indices"]),
            adapter_identity_sha256=value["adapter_identity_sha256"],
            trajectory_split_sha256=value["trajectory_split_sha256"],
            run_schema=value["run_schema"],
        )
        if result.bindings_sha256 != value.get("bindings_sha256"):
            raise Lfm2ClosedLoopRestartCapsuleError(
                "Capsule binding self-authentication failed."
            )
        return result


def lfm2_restart_capsule_adapter_identity(adapter: Any) -> dict[str, Any]:
    """Read and seal the adapter identities a restart capsule must pin."""

    callback = getattr(adapter, "restart_capsule_identity_bindings", None)
    if not callable(callback):
        raise Lfm2ClosedLoopRestartCapsuleError(
            "The adapter does not expose restart_capsule_identity_bindings."
        )
    identity = callback()
    if not isinstance(identity, Mapping):
        raise Lfm2ClosedLoopRestartCapsuleError(
            "Adapter restart identity bindings must be a mapping."
        )
    body = dict(identity)
    supplied = _digest(
        body.pop("adapter_identity_sha256", None),
        "adapter_identity_sha256",
    )
    if body.get("schema") != CLOSED_LOOP_RESTART_CAPSULE_BINDINGS_SCHEMA:
        raise Lfm2ClosedLoopRestartCapsuleError(
            "Adapter restart identity bindings carry the wrong schema."
        )
    if set(body) != {"schema", *_ADAPTER_IDENTITY_FIELDS}:
        raise Lfm2ClosedLoopRestartCapsuleError(
            "Adapter restart identity bindings have missing or unknown fields."
        )
    if canonical_sha256(body) != supplied:
        raise Lfm2ClosedLoopRestartCapsuleError(
            "Adapter restart identity binding seal is invalid."
        )
    return {**body, "adapter_identity_sha256": supplied}


def lfm2_restart_capsule_split_sha256(
    adapter: Any,
    config: ClosedLoopCompilerConfig,
) -> str:
    """Derive the deterministic split seal a writer must be bound to.

    The compiler seals a capsule against the exact fit/validation split it
    will use, but a writer has to exist before the compiler runs.  This helper
    recomputes that split from the same sealed adapter records and policy, so
    a caller never has to guess or duplicate the split rule.
    """

    if not isinstance(config, ClosedLoopCompilerConfig):
        raise TypeError("config must be ClosedLoopCompilerConfig.")
    records = tuple(adapter.parent_trajectory_records(ClosedLoopAudit()))
    return deterministic_whole_trajectory_split(
        records,
        validation_fraction=config.validation_fraction,
        split_seed=config.split_seed,
        selected_paths=config.selected_paths,
    ).split_sha256


class Lfm2DenseFoldProgressChain:
    """Read the hash-chained dense-fold progress position from a live sink.

    ``_Lfm2DenseFoldProgress`` lives in ``hf_lfm2_dense_budget_model_fold`` and
    exposes ``binding_sha256`` publicly but keeps its chain head and sequence
    private.  This reader duck-types both and fails closed when a supplied sink
    is not a dense-fold progress chain, so a capsule never records a fabricated
    chain position.
    """

    __slots__ = ("_sink", "_process_boundary")

    def __init__(self, sink: Any, *, process_boundary: bool = False) -> None:
        if sink is None:
            raise Lfm2ClosedLoopRestartCapsuleError(
                "A fold-progress chain view requires a progress sink."
            )
        _digest(
            getattr(sink, "binding_sha256", None),
            "fold progress binding_sha256",
        )
        self._sink = sink
        self._process_boundary = bool(process_boundary)

    def as_dict(self) -> dict[str, Any]:
        sink = self._sink
        binding = _digest(
            getattr(sink, "binding_sha256", None),
            "fold progress binding_sha256",
        )
        sequence = getattr(sink, "_sequence", None)
        previous = getattr(sink, "_previous_sha256", None)
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
        ):
            raise Lfm2ClosedLoopRestartCapsuleError(
                "The fold-progress sink does not expose a chain sequence."
            )
        body = {
            "schema": CLOSED_LOOP_RESTART_CAPSULE_PROGRESS_SCHEMA,
            "available": True,
            "binding_sha256": binding,
            "sequence": sequence,
            "previous_event_sha256": _digest(
                previous,
                "fold progress previous_event_sha256",
            ),
            "process_boundary": self._process_boundary,
        }
        return body


def unavailable_fold_progress_chain(
    *,
    process_boundary: bool = False,
) -> dict[str, Any]:
    """Record an explicitly absent fold-progress chain binding."""

    return {
        "schema": CLOSED_LOOP_RESTART_CAPSULE_PROGRESS_SCHEMA,
        "available": False,
        "binding_sha256": None,
        "sequence": None,
        "previous_event_sha256": None,
        "process_boundary": bool(process_boundary),
    }


def _validate_progress_chain(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "available",
        "binding_sha256",
        "sequence",
        "previous_event_sha256",
        "process_boundary",
    }:
        raise Lfm2ClosedLoopRestartCapsuleError(
            f"{label} fold-progress chain record is malformed."
        )
    result = dict(value)
    if (
        result["schema"] != CLOSED_LOOP_RESTART_CAPSULE_PROGRESS_SCHEMA
        or not isinstance(result["available"], bool)
        or not isinstance(result["process_boundary"], bool)
    ):
        raise Lfm2ClosedLoopRestartCapsuleError(
            f"{label} fold-progress chain record is malformed."
        )
    if result["available"]:
        _digest(result["binding_sha256"], f"{label} progress binding")
        _digest(result["previous_event_sha256"], f"{label} progress head")
        _nonnegative_int(result["sequence"], f"{label} progress sequence")
    elif any(
        result[name] is not None
        for name in ("binding_sha256", "sequence", "previous_event_sha256")
    ):
        raise Lfm2ClosedLoopRestartCapsuleError(
            f"{label} unavailable fold-progress chain carries chain values."
        )
    return result


@dataclass(frozen=True, slots=True)
class Lfm2ClosedLoopRestartCapsuleReference:
    """External pins required to discover and authenticate one capsule."""

    path: Path
    capsule_key: str
    manifest_sha256: str
    manifest_file_sha256: str
    bindings_sha256: str
    child_state_sha256: str
    capsule_chain_sha256: str
    capsule_sequence: int
    layer_index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).expanduser().absolute())
        for name in (
            "capsule_key",
            "manifest_sha256",
            "manifest_file_sha256",
            "bindings_sha256",
            "child_state_sha256",
            "capsule_chain_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "capsule_sequence",
            _nonnegative_int(self.capsule_sequence, "capsule_sequence"),
        )
        object.__setattr__(
            self,
            "layer_index",
            _nonnegative_int(self.layer_index, "layer_index"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "capsule_key": self.capsule_key,
            "manifest_sha256": self.manifest_sha256,
            "manifest_file_sha256": self.manifest_file_sha256,
            "bindings_sha256": self.bindings_sha256,
            "child_state_sha256": self.child_state_sha256,
            "capsule_chain_sha256": self.capsule_chain_sha256,
            "capsule_sequence": self.capsule_sequence,
            "layer_index": self.layer_index,
        }


@dataclass(frozen=True, slots=True)
class Lfm2ClosedLoopRestartCapsuleAuthentication:
    """One fully authenticated capsule, without materializing its tensors."""

    path: Path
    reference: Lfm2ClosedLoopRestartCapsuleReference
    bindings: Lfm2ClosedLoopRestartCapsuleBindings
    manifest: Mapping[str, Any]
    completed_layer_indices: tuple[int, ...]
    resume_layer_index: int | None
    capsule_phase: str = _ACCEPTED_LAYER_PHASE
    terminal_phase: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class Lfm2ClosedLoopRestartCapsuleResume:
    """A freshly loaded child plus the exact resume point it authenticates."""

    path: Path
    reference: Lfm2ClosedLoopRestartCapsuleReference
    bindings: Lfm2ClosedLoopRestartCapsuleBindings
    manifest: Mapping[str, Any]
    completed_layer_indices: tuple[int, ...]
    resume_layer_index: int | None
    child_state_sha256: str
    child_topology_sha256: str
    validation_decision_sha256: str
    capsule_phase: str = _ACCEPTED_LAYER_PHASE
    terminal_phase: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        body = {
            "schema": CLOSED_LOOP_RESTART_CAPSULE_RESUME_SCHEMA,
            "path": str(self.path),
            "capsule_key": self.reference.capsule_key,
            "capsule_sequence": self.reference.capsule_sequence,
            "capsule_chain_sha256": self.reference.capsule_chain_sha256,
            "manifest_sha256": self.reference.manifest_sha256,
            "bindings_sha256": self.bindings.bindings_sha256,
            "completed_layer_indices": list(self.completed_layer_indices),
            "resume_layer_index": self.resume_layer_index,
            "child_state_sha256": self.child_state_sha256,
            "child_topology_sha256": self.child_topology_sha256,
            "validation_decision_sha256": self.validation_decision_sha256,
            "child_state_restored_into_a_fresh_child": True,
            "skipped_layers_are_not_replayed": True,
        }
        if self.capsule_phase != _ACCEPTED_LAYER_PHASE:
            # An accepted-layer resume keeps its historical sealed body.
            body = {
                **body,
                "capsule_phase": self.capsule_phase,
                "terminal_phase": dict(self.terminal_phase or {}),
            }
        return {**body, "resume_sha256": canonical_sha256(body)}


def _capsule_directory_name(*, sequence: int, capsule_key: str) -> str:
    return f"{_DIRECTORY_PREFIX}{sequence:06d}-{capsule_key}"


def _root_directory(value: str | os.PathLike[str]) -> Path:
    root = Path(value).expanduser().absolute()
    if _is_link_or_junction(root):
        raise Lfm2ClosedLoopRestartCapsuleError(
            "The restart capsule root cannot be a link or junction."
        )
    _real_directory(root, name="restart capsule root")
    if root.resolve(strict=True) != root:
        raise Lfm2ClosedLoopRestartCapsuleError(
            "The restart capsule root path cannot traverse a link."
        )
    return root


def _capsule_root_path(value: str | os.PathLike[str]) -> tuple[Path, tuple[int, int]]:
    lexical = Path(value).expanduser().absolute()
    if _is_link_or_junction(lexical):
        raise Lfm2ClosedLoopRestartCapsuleError(
            "A restart capsule cannot be a link or junction."
        )
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise Lfm2ClosedLoopRestartCapsuleError(
            "The restart capsule is missing."
        ) from exc
    if resolved != lexical:
        raise Lfm2ClosedLoopRestartCapsuleError(
            "A restart capsule path cannot traverse a link."
        )
    root_stat = _real_directory(resolved, name="restart capsule")
    try:
        entries = tuple(os.scandir(resolved))
    except OSError as exc:
        raise Lfm2ClosedLoopRestartCapsuleError(
            "The restart capsule cannot be enumerated."
        ) from exc
    if {entry.name for entry in entries} != _ROOT_INVENTORY:
        raise Lfm2ClosedLoopRestartCapsuleError(
            "The restart capsule has missing or unexpected entries."
        )
    for entry in entries:
        _regular_file(resolved / entry.name, name=entry.name)
    return resolved, _stat_identity(root_stat)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_remove_stage(
    stage: Path,
    *,
    parent: Path,
    identity: tuple[int, int],
) -> None:
    if not _lexists(stage):
        return
    current = stage.lstat()
    if (
        _stat_identity(current) != identity
        or not stat.S_ISDIR(current.st_mode)
        or _is_link_or_junction(stage)
        or stage.resolve(strict=True).parent != parent.resolve(strict=True)
        or not stage.name.startswith(_STAGE_PREFIX)
    ):
        raise Lfm2ClosedLoopRestartCapsuleError(
            "Refusing to remove unauthenticated restart capsule staging."
        )
    shutil.rmtree(stage)
    _fsync_directory(parent)


def lfm2_restart_capsule_terminal_phase(
    *,
    body_reclosure_enabled: bool,
    body_reclosure_sha256: str | None,
    final_normalization_sha256: str,
    output_interface_enabled: bool,
    output_interface_sha256: str | None,
) -> dict[str, Any]:
    """Seal what the terminal capsule proves has already been solved.

    A terminal capsule is taken after the output interface and before any
    evaluation, so a crash in the evaluators no longer costs the body, the
    reclosure, the final normalization, or the output-interface solves.  It
    pins the sealed digest of each of those stage reports, so a resumed process
    can prove it is continuing the same transaction rather than a similar one.
    """

    if not isinstance(body_reclosure_enabled, bool) or not isinstance(
        output_interface_enabled, bool
    ):
        raise Lfm2ClosedLoopRestartCapsuleError(
            "Terminal-phase stage flags must be boolean."
        )
    body = {
        "schema": CLOSED_LOOP_RESTART_CAPSULE_TERMINAL_PHASE_SCHEMA,
        "body_reclosure_enabled": body_reclosure_enabled,
        "body_reclosure_sha256": _optional_digest(
            body_reclosure_sha256,
            "body_reclosure_sha256",
        ),
        "final_normalization_sha256": _digest(
            final_normalization_sha256,
            "final_normalization_sha256",
        ),
        "output_interface_enabled": output_interface_enabled,
        "output_interface_sha256": _optional_digest(
            output_interface_sha256,
            "output_interface_sha256",
        ),
        "captured_after_output_interface": True,
        "captured_before_any_evaluation": True,
        "remaining_stage": "evaluation",
        "body_layers_are_never_replayed_on_resume": True,
        "terminal_solves_are_never_replayed_on_resume": True,
    }
    return {**body, "terminal_phase_sha256": canonical_sha256(body)}


def _validate_terminal_phase(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2ClosedLoopRestartCapsuleError(
            "A terminal restart capsule requires its sealed terminal phase."
        )
    body = dict(value)
    supplied = body.pop("terminal_phase_sha256", None)
    if (
        body.get("schema") != CLOSED_LOOP_RESTART_CAPSULE_TERMINAL_PHASE_SCHEMA
        or body.get("captured_after_output_interface") is not True
        or body.get("captured_before_any_evaluation") is not True
        or body.get("remaining_stage") != "evaluation"
        or canonical_sha256(body) != supplied
    ):
        raise Lfm2ClosedLoopRestartCapsuleError(
            "The terminal restart-capsule phase seal is invalid."
        )
    return dict(value)


def _capsule_body(
    *,
    bindings: Lfm2ClosedLoopRestartCapsuleBindings,
    adapter_identity: Mapping[str, Any],
    layer_index: int,
    layer_ordinal: int,
    completed_layer_indices: tuple[int, ...],
    remaining_layer_indices: tuple[int, ...],
    validation_decision_sha256: str,
    child_state_sha256: str,
    child_topology_sha256: str,
    origin_process_local_child_state_sha256: str,
    origin_process_local_child_topology_sha256: str,
    child_state_alias_groups_value: tuple[tuple[str, ...], ...],
    child_state_file: Mapping[str, Any],
    capsule_sequence: int,
    previous_capsule_manifest_sha256: str,
    previous_capsule_chain_sha256: str,
    resumed_from_capsule_manifest_sha256: str | None,
    fold_progress: Mapping[str, Any],
    capsule_phase: str = _ACCEPTED_LAYER_PHASE,
    terminal_phase: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if capsule_phase == _ACCEPTED_LAYER_PHASE:
        # Pinned byte-for-byte: every capsule sealed before the terminal phase
        # existed keeps its content key, chain, and manifest digest.
        phase_fields: dict[str, Any] = {}
    else:
        phase_fields = {
            "capsule_phase": capsule_phase,
            "terminal_phase": dict(terminal_phase or {}),
        }
    return {
        "schema": CLOSED_LOOP_RESTART_CAPSULE_SCHEMA,
        "artifact_kind": _ARTIFACT_KIND,
        **_NONPROMOTABLE_FLAGS,
        **phase_fields,
        "bindings": bindings.as_dict(),
        "bindings_sha256": bindings.bindings_sha256,
        "adapter_identity": dict(adapter_identity),
        "layer_index": layer_index,
        "layer_ordinal": layer_ordinal,
        "completed_layer_indices": list(completed_layer_indices),
        "remaining_layer_indices": list(remaining_layer_indices),
        "resume_layer_index": (
            remaining_layer_indices[0] if remaining_layer_indices else None
        ),
        "validation_decision_sha256": validation_decision_sha256,
        "child_state_sha256": child_state_sha256,
        "child_topology_sha256": child_topology_sha256,
        "origin_process_local_child_state_sha256": (
            origin_process_local_child_state_sha256
        ),
        "origin_process_local_child_topology_sha256": (
            origin_process_local_child_topology_sha256
        ),
        "process_local_digests_are_not_portable": True,
        "child_state_alias_groups": [
            list(names) for names in child_state_alias_groups_value
        ],
        "child_state_file": dict(child_state_file),
        "capsule_sequence": capsule_sequence,
        "previous_capsule_manifest_sha256": previous_capsule_manifest_sha256,
        "previous_capsule_chain_sha256": previous_capsule_chain_sha256,
        "resumed_from_capsule_manifest_sha256": (
            resumed_from_capsule_manifest_sha256
        ),
        "fold_progress": dict(fold_progress),
        "accepted_layer_validation_sealed": True,
        "skipped_layers_are_never_replayed_on_resume": True,
        "operation_audit": dict(_ZERO_OPERATION_AUDIT),
    }


def _capsule_key(body: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "schema": CLOSED_LOOP_RESTART_CAPSULE_KEY_SCHEMA,
            "capsule_body": dict(body),
        }
    )


def _capsule_chain_sha256(
    *,
    sequence: int,
    previous_capsule_chain_sha256: str,
    capsule_key: str,
) -> str:
    return canonical_sha256(
        {
            "schema": CLOSED_LOOP_RESTART_CAPSULE_CHAIN_SCHEMA,
            "sequence": sequence,
            "previous_capsule_chain_sha256": previous_capsule_chain_sha256,
            "capsule_key": capsule_key,
        }
    )


def _open_capsule(
    reference: Lfm2ClosedLoopRestartCapsuleReference,
) -> dict[str, Any]:
    """Authenticate every capsule byte, seal, and chain field."""

    if not isinstance(reference, Lfm2ClosedLoopRestartCapsuleReference):
        raise TypeError(
            "reference must be Lfm2ClosedLoopRestartCapsuleReference."
        )
    root, root_identity = _capsule_root_path(reference.path)
    manifest, manifest_raw = _read_canonical_json_with_raw(
        root / _MANIFEST_NAME,
        name=_MANIFEST_NAME,
    )
    manifest = _verify_seal(
        manifest,
        field_name="manifest_sha256",
        label="restart capsule manifest",
    )
    _validate_nonpromotable(manifest, label="restart capsule manifest")
    body = dict(manifest)
    directory_name = body.pop("directory_name", None)
    capsule_key = _digest(body.pop("capsule_key", None), "capsule_key")
    chain_sha256 = _digest(
        body.pop("capsule_chain_sha256", None),
        "capsule_chain_sha256",
    )
    body.pop("manifest_sha256", None)
    if (
        manifest.get("schema") != CLOSED_LOOP_RESTART_CAPSULE_SCHEMA
        or manifest.get("artifact_kind") != _ARTIFACT_KIND
        or manifest.get("operation_audit") != _ZERO_OPERATION_AUDIT
        or manifest.get("accepted_layer_validation_sealed") is not True
        or manifest.get("skipped_layers_are_never_replayed_on_resume")
        is not True
        or manifest.get("process_local_digests_are_not_portable") is not True
        or _capsule_key(body) != capsule_key
        or directory_name != root.name
    ):
        raise Lfm2ClosedLoopRestartCapsuleError(
            "Restart capsule manifest contract or content key is invalid."
        )
    sequence = _nonnegative_int(
        manifest.get("capsule_sequence"),
        "capsule_sequence",
    )
    previous_chain = _digest(
        manifest.get("previous_capsule_chain_sha256"),
        "previous_capsule_chain_sha256",
    )
    _digest(
        manifest.get("previous_capsule_manifest_sha256"),
        "previous_capsule_manifest_sha256",
    )
    _optional_digest(
        manifest.get("resumed_from_capsule_manifest_sha256"),
        "resumed_from_capsule_manifest_sha256",
    )
    if _capsule_chain_sha256(
        sequence=sequence,
        previous_capsule_chain_sha256=previous_chain,
        capsule_key=capsule_key,
    ) != chain_sha256:
        raise Lfm2ClosedLoopRestartCapsuleError(
            "Restart capsule chain seal is invalid."
        )
    if sequence == 0 and previous_chain != (
        CLOSED_LOOP_RESTART_CAPSULE_CHAIN_GENESIS_SHA256
    ):
        raise Lfm2ClosedLoopRestartCapsuleError(
            "The first restart capsule must chain to genesis."
        )
    _validate_progress_chain(
        manifest.get("fold_progress"),
        label="restart capsule",
    )
    bindings = Lfm2ClosedLoopRestartCapsuleBindings.from_mapping(
        manifest.get("bindings")
    )
    adapter_identity = manifest.get("adapter_identity")
    if not isinstance(adapter_identity, Mapping):
        raise Lfm2ClosedLoopRestartCapsuleError(
            "Restart capsule adapter identity is missing."
        )
    identity_body = dict(adapter_identity)
    identity_sha256 = _digest(
        identity_body.pop("adapter_identity_sha256", None),
        "adapter_identity_sha256",
    )
    if (
        set(identity_body) != {"schema", *_ADAPTER_IDENTITY_FIELDS}
        or identity_body.get("schema")
        != CLOSED_LOOP_RESTART_CAPSULE_BINDINGS_SCHEMA
        or canonical_sha256(identity_body) != identity_sha256
        or identity_sha256 != bindings.adapter_identity_sha256
    ):
        raise Lfm2ClosedLoopRestartCapsuleError(
            "Restart capsule adapter identity seal is invalid."
        )
    layer_index = _nonnegative_int(manifest.get("layer_index"), "layer_index")
    completed = _layer_indices(
        manifest.get("completed_layer_indices"),
        "completed_layer_indices",
    )
    remaining = _layer_indices(
        manifest.get("remaining_layer_indices"),
        "remaining_layer_indices",
    )
    resume_layer_index = manifest.get("resume_layer_index")
    if (
        layer_index not in completed
        or set(completed) & set(remaining)
        or not set(completed) | set(remaining)
        <= set(bindings.selected_layer_indices)
        or resume_layer_index != (remaining[0] if remaining else None)
        or len(completed) != len(set(completed))
    ):
        raise Lfm2ClosedLoopRestartCapsuleError(
            "Restart capsule layer inventory is inconsistent."
        )
    capsule_phase = manifest.get("capsule_phase", _ACCEPTED_LAYER_PHASE)
    terminal_phase = None
    if capsule_phase not in _CAPSULE_PHASES:
        raise Lfm2ClosedLoopRestartCapsuleError(
            "Restart capsule phase is unknown."
        )
    if capsule_phase == _ACCEPTED_LAYER_PHASE:
        if "terminal_phase" in manifest:
            raise Lfm2ClosedLoopRestartCapsuleError(
                "An accepted-layer capsule cannot carry a terminal phase."
            )
    else:
        terminal_phase = _validate_terminal_phase(manifest.get("terminal_phase"))
        if remaining or set(completed) != set(bindings.selected_layer_indices):
            raise Lfm2ClosedLoopRestartCapsuleError(
                "A terminal restart capsule must follow every selected layer."
            )
    child_state_sha256 = _digest(
        manifest.get("child_state_sha256"),
        "child_state_sha256",
    )
    _digest(manifest.get("child_topology_sha256"), "child_topology_sha256")
    _digest(
        manifest.get("validation_decision_sha256"),
        "validation_decision_sha256",
    )
    record = manifest.get("child_state_file")
    if not isinstance(record, Mapping) or record.get("path") != (
        _CHILD_STATE_NAME
    ):
        raise Lfm2ClosedLoopRestartCapsuleError(
            "Restart capsule child-state file record is invalid."
        )
    manifest_file_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    if (
        manifest_file_sha256 != reference.manifest_file_sha256
        or manifest.get("manifest_sha256") != reference.manifest_sha256
        or capsule_key != reference.capsule_key
        or chain_sha256 != reference.capsule_chain_sha256
        or bindings.bindings_sha256 != reference.bindings_sha256
        or child_state_sha256 != reference.child_state_sha256
        or sequence != reference.capsule_sequence
        or layer_index != reference.layer_index
    ):
        raise Lfm2ClosedLoopRestartCapsuleError(
            "Restart capsule differs from its external pins."
        )
    _load_safetensors_path(
        root / _CHILD_STATE_NAME,
        name=_CHILD_STATE_NAME,
        artifact_kind=_ARTIFACT_KIND,
        file_role=_CHILD_STATE_FILE_ROLE,
        record=record,
        clone_tensors=False,
    )
    current_root, current_identity = _capsule_root_path(root)
    if current_root != root or current_identity != root_identity:
        raise Lfm2ClosedLoopRestartCapsuleError(
            "The restart capsule changed during authentication."
        )
    return {
        "root": root,
        "manifest": manifest,
        "bindings": bindings,
        "completed_layer_indices": completed,
        "resume_layer_index": resume_layer_index,
        "child_state_record": record,
        "capsule_phase": capsule_phase,
        "terminal_phase": terminal_phase,
    }


def authenticate_lfm2_closed_loop_restart_capsule(
    reference: Lfm2ClosedLoopRestartCapsuleReference,
) -> Lfm2ClosedLoopRestartCapsuleAuthentication:
    """Authenticate one capsule without materializing a child model."""

    opened = _open_capsule(reference)
    terminal_phase = opened["terminal_phase"]
    return Lfm2ClosedLoopRestartCapsuleAuthentication(
        path=opened["root"],
        reference=reference,
        bindings=opened["bindings"],
        manifest=MappingProxyType(dict(opened["manifest"])),
        completed_layer_indices=opened["completed_layer_indices"],
        resume_layer_index=opened["resume_layer_index"],
        capsule_phase=opened["capsule_phase"],
        terminal_phase=(
            None if terminal_phase is None else MappingProxyType(terminal_phase)
        ),
    )


def _reference_from_manifest(
    root: Path,
    manifest: Mapping[str, Any],
    manifest_file_sha256: str,
) -> Lfm2ClosedLoopRestartCapsuleReference:
    bindings = manifest.get("bindings")
    if not isinstance(bindings, Mapping):
        raise Lfm2ClosedLoopRestartCapsuleError(
            "Restart capsule manifest has no bindings."
        )
    return Lfm2ClosedLoopRestartCapsuleReference(
        path=root,
        capsule_key=manifest.get("capsule_key"),
        manifest_sha256=manifest.get("manifest_sha256"),
        manifest_file_sha256=manifest_file_sha256,
        bindings_sha256=bindings.get("bindings_sha256"),
        child_state_sha256=manifest.get("child_state_sha256"),
        capsule_chain_sha256=manifest.get("capsule_chain_sha256"),
        capsule_sequence=manifest.get("capsule_sequence"),
        layer_index=manifest.get("layer_index"),
    )


def discover_lfm2_closed_loop_restart_capsules(
    directory: str | os.PathLike[str],
) -> tuple[Lfm2ClosedLoopRestartCapsuleReference, ...]:
    """Authenticate every capsule under one root, ascending by sequence.

    Rotation always retains a contiguous newest run of capsules, so a gap in
    the discovered sequence numbers means a capsule was removed or forged.
    That fails closed rather than silently resuming from a partial chain.
    """

    root = _root_directory(directory)
    try:
        entries = tuple(sorted(os.scandir(root), key=lambda item: item.name))
    except OSError as exc:
        raise Lfm2ClosedLoopRestartCapsuleError(
            "The restart capsule root cannot be enumerated."
        ) from exc
    references: list[Lfm2ClosedLoopRestartCapsuleReference] = []
    authentications: list[Lfm2ClosedLoopRestartCapsuleAuthentication] = []
    for entry in entries:
        if entry.name.startswith(_STAGE_PREFIX):
            continue
        path = root / entry.name
        if not entry.name.startswith(_DIRECTORY_PREFIX):
            raise Lfm2ClosedLoopRestartCapsuleError(
                "The restart capsule root contains a foreign entry."
            )
        capsule_root, _identity = _capsule_root_path(path)
        manifest, manifest_raw = _read_canonical_json_with_raw(
            capsule_root / _MANIFEST_NAME,
            name=_MANIFEST_NAME,
        )
        reference = _reference_from_manifest(
            capsule_root,
            manifest,
            hashlib.sha256(manifest_raw).hexdigest(),
        )
        authentication = authenticate_lfm2_closed_loop_restart_capsule(
            reference
        )
        references.append(reference)
        authentications.append(authentication)
    order = sorted(
        range(len(references)),
        key=lambda index: references[index].capsule_sequence,
    )
    references = [references[index] for index in order]
    authentications = [authentications[index] for index in order]
    sequences = [reference.capsule_sequence for reference in references]
    if len(set(sequences)) != len(sequences) or any(
        second - first != 1
        for first, second in zip(sequences, sequences[1:], strict=False)
    ):
        raise Lfm2ClosedLoopRestartCapsuleError(
            "Restart capsule sequence numbers are duplicated or discontiguous."
        )
    for previous, current in zip(
        authentications, authentications[1:], strict=False
    ):
        previous_manifest = previous.manifest
        current_manifest = current.manifest
        if (
            current_manifest["previous_capsule_manifest_sha256"]
            != previous_manifest["manifest_sha256"]
            or current_manifest["previous_capsule_chain_sha256"]
            != previous_manifest["capsule_chain_sha256"]
            or previous.bindings.bindings_sha256
            != current.bindings.bindings_sha256
        ):
            raise Lfm2ClosedLoopRestartCapsuleError(
                "Restart capsule chain continuity is broken."
            )
        previous_progress = previous_manifest["fold_progress"]
        current_progress = current_manifest["fold_progress"]
        both_absent = not (
            previous_progress["available"] or current_progress["available"]
        )
        same_process = (
            previous_progress["available"]
            and current_progress["available"]
            and previous_progress["binding_sha256"]
            == current_progress["binding_sha256"]
        )
        if same_process:
            if current_progress["sequence"] <= previous_progress["sequence"]:
                raise Lfm2ClosedLoopRestartCapsuleError(
                    "Restart capsule fold-progress chain did not advance."
                )
        elif not both_absent and not current_progress["process_boundary"]:
            raise Lfm2ClosedLoopRestartCapsuleError(
                "A restart capsule changed fold-progress chains without "
                "declaring a process boundary."
            )
    return tuple(references)


def latest_lfm2_closed_loop_restart_capsule(
    directory: str | os.PathLike[str],
) -> Lfm2ClosedLoopRestartCapsuleReference | None:
    """Return the newest authenticated capsule, or None when none exists."""

    references = discover_lfm2_closed_loop_restart_capsules(directory)
    return references[-1] if references else None


def _install_child_state(
    model: Any,
    tensors: Mapping[str, Any],
    alias_groups: Sequence[Sequence[str]],
) -> None:
    """Copy sealed tensors into a fresh child without changing any binding."""

    torch = _torch()
    state = _child_state_dict(model)
    live_groups = child_state_alias_groups(model)
    expected_groups = tuple(tuple(str(name) for name in names) for names in alias_groups)
    if live_groups != expected_groups:
        raise Lfm2ClosedLoopRestartCapsuleError(
            "The fresh child does not share the sealed tensor alias grouping."
        )
    if set(tensors) != {names[0] for names in expected_groups}:
        raise Lfm2ClosedLoopRestartCapsuleError(
            "Sealed capsule tensors do not cover exactly one alias per group."
        )
    for names in expected_groups:
        source = tensors[names[0]]
        for name in names:
            target = state[name]
            if (
                tuple(target.shape) != tuple(source.shape)
                or target.dtype != source.dtype
            ):
                raise Lfm2ClosedLoopRestartCapsuleError(
                    f"Child state entry {name!r} changed shape or dtype."
                )
    with torch.no_grad():
        for names in expected_groups:
            source = tensors[names[0]]
            for name in names:
                target = state[name]
                target.copy_(
                    source.to(device=target.device, dtype=target.dtype)
                )
    if any(
        parameter.grad is not None
        for _name, parameter in model.named_parameters(remove_duplicate=False)
    ):
        raise Lfm2ClosedLoopRestartCapsuleError(
            "A resumed child must be gradient-free."
        )


def load_lfm2_closed_loop_restart_capsule(
    reference: Lfm2ClosedLoopRestartCapsuleReference,
    *,
    child_model: Any,
    expected_bindings: Lfm2ClosedLoopRestartCapsuleBindings | None = None,
    expected_adapter_identity: Mapping[str, Any] | None = None,
) -> Lfm2ClosedLoopRestartCapsuleResume:
    """Authenticate one capsule and restore it into a freshly provided child."""

    opened = _open_capsule(reference)
    root = opened["root"]
    manifest = opened["manifest"]
    bindings = opened["bindings"]
    if expected_bindings is not None:
        if not isinstance(
            expected_bindings, Lfm2ClosedLoopRestartCapsuleBindings
        ):
            raise TypeError(
                "expected_bindings must be "
                "Lfm2ClosedLoopRestartCapsuleBindings."
            )
        if expected_bindings.bindings_sha256 != bindings.bindings_sha256:
            raise Lfm2ClosedLoopRestartCapsuleError(
                "The restart capsule was sealed for a different run identity."
            )
    if expected_adapter_identity is not None:
        expected_identity = dict(expected_adapter_identity)
        if expected_identity != dict(manifest["adapter_identity"]):
            raise Lfm2ClosedLoopRestartCapsuleError(
                "The restart capsule adapter identity differs from the live "
                "adapter."
            )
    tensors = _load_safetensors_path(
        root / _CHILD_STATE_NAME,
        name=_CHILD_STATE_NAME,
        artifact_kind=_ARTIFACT_KIND,
        file_role=_CHILD_STATE_FILE_ROLE,
        record=opened["child_state_record"],
        clone_tensors=True,
    )
    _install_child_state(
        child_model,
        tensors,
        manifest["child_state_alias_groups"],
    )
    restored_state = portable_child_state_sha256(child_model)
    restored_topology = portable_child_topology_sha256(child_model)
    if (
        restored_state != manifest["child_state_sha256"]
        or restored_topology != manifest["child_topology_sha256"]
    ):
        raise Lfm2ClosedLoopRestartCapsuleError(
            "The restored child does not reproduce the sealed capsule state."
        )
    terminal_phase = opened["terminal_phase"]
    return Lfm2ClosedLoopRestartCapsuleResume(
        path=root,
        reference=reference,
        bindings=bindings,
        manifest=MappingProxyType(dict(manifest)),
        completed_layer_indices=opened["completed_layer_indices"],
        resume_layer_index=opened["resume_layer_index"],
        child_state_sha256=restored_state,
        child_topology_sha256=restored_topology,
        validation_decision_sha256=str(
            manifest["validation_decision_sha256"]
        ),
        capsule_phase=opened["capsule_phase"],
        terminal_phase=(
            None if terminal_phase is None else MappingProxyType(terminal_phase)
        ),
    )


def publish_lfm2_closed_loop_restart_capsule(
    *,
    child_model: Any,
    directory: str | os.PathLike[str],
    bindings: Lfm2ClosedLoopRestartCapsuleBindings,
    adapter_identity: Mapping[str, Any],
    layer_index: int,
    layer_ordinal: int,
    completed_layer_indices: Sequence[int],
    remaining_layer_indices: Sequence[int],
    validation_decision_sha256: str,
    origin_process_local_child_state_sha256: str,
    origin_process_local_child_topology_sha256: str,
    capsule_sequence: int,
    previous_capsule_manifest_sha256: str,
    previous_capsule_chain_sha256: str,
    resumed_from_capsule_manifest_sha256: str | None,
    fold_progress: Mapping[str, Any],
    capsule_phase: str = _ACCEPTED_LAYER_PHASE,
    terminal_phase: Mapping[str, Any] | None = None,
) -> Lfm2ClosedLoopRestartCapsuleReference:
    """Atomically publish one sealed accepted-layer or terminal restart point."""

    if not isinstance(bindings, Lfm2ClosedLoopRestartCapsuleBindings):
        raise TypeError(
            "bindings must be Lfm2ClosedLoopRestartCapsuleBindings."
        )
    phase = _text(capsule_phase, "capsule_phase")
    if phase not in _CAPSULE_PHASES:
        raise Lfm2ClosedLoopRestartCapsuleError(
            "capsule_phase must be accepted_body_layer or "
            "terminal_post_output_interface."
        )
    if (terminal_phase is not None) != (phase == _TERMINAL_PHASE):
        raise Lfm2ClosedLoopRestartCapsuleError(
            "Only a terminal restart capsule carries a sealed terminal phase."
        )
    normalized_terminal_phase = (
        None if terminal_phase is None else _validate_terminal_phase(terminal_phase)
    )
    root = _root_directory(directory)
    completed = _layer_indices(
        completed_layer_indices,
        "completed_layer_indices",
    )
    remaining = _layer_indices(
        remaining_layer_indices,
        "remaining_layer_indices",
    )
    layer_index = _nonnegative_int(layer_index, "layer_index")
    if layer_index not in completed or set(completed) & set(remaining):
        raise Lfm2ClosedLoopRestartCapsuleError(
            "A capsule must record its own accepted layer exactly once."
        )
    progress = _validate_progress_chain(
        fold_progress,
        label="restart capsule",
    )
    state_tensors = _child_state_dict(child_model)
    groups = child_state_alias_groups(child_model)
    canonical = {names[0]: state_tensors[names[0]] for names in groups}
    normalized, _entries = _normalize_tensors(
        canonical,
        name="closed-loop restart child state",
        clone=True,
    )
    child_state_sha256 = portable_child_state_sha256(child_model)
    child_topology_sha256 = portable_child_topology_sha256(child_model)
    stage = Path(
        tempfile.mkdtemp(prefix=_STAGE_PREFIX, dir=str(root))
    ).absolute()
    stage_identity = _stat_identity(stage.lstat())
    moved = False
    destination: Path | None = None
    try:
        state_path = stage / _CHILD_STATE_NAME
        _save_safetensors(
            state_path,
            normalized,
            artifact_kind=_ARTIFACT_KIND,
            file_role=_CHILD_STATE_FILE_ROLE,
        )
        state_record = _file_record(state_path, tensor_entries=_entries)
        body = _capsule_body(
            bindings=bindings,
            adapter_identity=adapter_identity,
            layer_index=layer_index,
            layer_ordinal=_nonnegative_int(layer_ordinal, "layer_ordinal"),
            completed_layer_indices=completed,
            remaining_layer_indices=remaining,
            validation_decision_sha256=_digest(
                validation_decision_sha256,
                "validation_decision_sha256",
            ),
            child_state_sha256=child_state_sha256,
            child_topology_sha256=child_topology_sha256,
            origin_process_local_child_state_sha256=_digest(
                origin_process_local_child_state_sha256,
                "origin_process_local_child_state_sha256",
            ),
            origin_process_local_child_topology_sha256=_digest(
                origin_process_local_child_topology_sha256,
                "origin_process_local_child_topology_sha256",
            ),
            child_state_alias_groups_value=groups,
            child_state_file=state_record,
            capsule_sequence=_nonnegative_int(
                capsule_sequence,
                "capsule_sequence",
            ),
            previous_capsule_manifest_sha256=_digest(
                previous_capsule_manifest_sha256,
                "previous_capsule_manifest_sha256",
            ),
            previous_capsule_chain_sha256=_digest(
                previous_capsule_chain_sha256,
                "previous_capsule_chain_sha256",
            ),
            resumed_from_capsule_manifest_sha256=_optional_digest(
                resumed_from_capsule_manifest_sha256,
                "resumed_from_capsule_manifest_sha256",
            ),
            fold_progress=progress,
            capsule_phase=phase,
            terminal_phase=normalized_terminal_phase,
        )
        capsule_key = _capsule_key(body)
        directory_name = _capsule_directory_name(
            sequence=int(capsule_sequence),
            capsule_key=capsule_key,
        )
        manifest = _sealed(
            {
                **body,
                "capsule_key": capsule_key,
                "capsule_chain_sha256": _capsule_chain_sha256(
                    sequence=int(capsule_sequence),
                    previous_capsule_chain_sha256=(
                        previous_capsule_chain_sha256
                    ),
                    capsule_key=capsule_key,
                ),
                "directory_name": directory_name,
            },
            "manifest_sha256",
        )
        manifest_path = stage / _MANIFEST_NAME
        manifest_raw = _write_canonical_json(manifest_path, manifest)
        _fsync_directory(stage)
        destination = root / directory_name
        staged_reference = Lfm2ClosedLoopRestartCapsuleReference(
            path=stage,
            capsule_key=capsule_key,
            manifest_sha256=manifest["manifest_sha256"],
            manifest_file_sha256=hashlib.sha256(manifest_raw).hexdigest(),
            bindings_sha256=bindings.bindings_sha256,
            child_state_sha256=child_state_sha256,
            capsule_chain_sha256=manifest["capsule_chain_sha256"],
            capsule_sequence=int(capsule_sequence),
            layer_index=layer_index,
        )
        # The staged manifest pins its published directory name, so it can only
        # be authenticated after the atomic rename.
        if _lexists(destination):
            raise Lfm2ClosedLoopRestartCapsuleError(
                f"A restart capsule already occupies {destination}."
            )
        _atomic_rename_noreplace(stage, destination)
        moved = True
        _fsync_directory(root)
        reference = Lfm2ClosedLoopRestartCapsuleReference(
            path=destination,
            capsule_key=staged_reference.capsule_key,
            manifest_sha256=staged_reference.manifest_sha256,
            manifest_file_sha256=staged_reference.manifest_file_sha256,
            bindings_sha256=staged_reference.bindings_sha256,
            child_state_sha256=staged_reference.child_state_sha256,
            capsule_chain_sha256=staged_reference.capsule_chain_sha256,
            capsule_sequence=staged_reference.capsule_sequence,
            layer_index=staged_reference.layer_index,
        )
        _open_capsule(reference)
        return reference
    except FileExistsError as exc:
        raise Lfm2ClosedLoopRestartCapsuleError(
            f"Restart capsule destination already exists: {destination}."
        ) from exc
    finally:
        if not moved and _lexists(stage):
            _safe_remove_stage(stage, parent=root, identity=stage_identity)


def rotate_lfm2_closed_loop_restart_capsules(
    directory: str | os.PathLike[str],
    *,
    keep: int = DEFAULT_RESTART_CAPSULE_ROTATION_KEEP,
) -> tuple[Lfm2ClosedLoopRestartCapsuleReference, ...]:
    """Keep only the newest ``keep`` authenticated capsules.

    Only fully authenticated capsule directories under the supplied root are
    removed, the newest capsule is never removed, and the retained set stays a
    contiguous newest run so the chain remains verifiable.
    """

    keep = _positive_int(keep, "keep")
    root = _root_directory(directory)
    references = discover_lfm2_closed_loop_restart_capsules(root)
    if len(references) <= keep:
        return references
    retired = references[: len(references) - keep]
    retained = references[len(references) - keep:]
    for reference in retired:
        path = reference.path
        if (
            path.parent != root
            or not path.name.startswith(_DIRECTORY_PREFIX)
            or _is_link_or_junction(path)
        ):
            raise Lfm2ClosedLoopRestartCapsuleError(
                "Refusing to rotate an unauthenticated restart capsule path."
            )
        _capsule_root_path(path)
        shutil.rmtree(path)
    _fsync_directory(root)
    remaining = discover_lfm2_closed_loop_restart_capsules(root)
    if tuple(value.capsule_key for value in remaining) != tuple(
        value.capsule_key for value in retained
    ):
        raise Lfm2ClosedLoopRestartCapsuleError(
            "Restart capsule rotation did not retain the newest capsules."
        )
    return remaining


class Lfm2ClosedLoopRestartCapsuleWriter:
    """Accepted-layer restart capsule writer for ``ChildClosedLoopCompiler``.

    The writer satisfies the orchestrator's family-neutral
    ``AcceptedLayerRestartCapsuleWriter`` protocol.  It publishes one sealed
    capsule after every accepted layer, rotates to a bounded number of
    capsules, and reports which layers a prior process already accepted.
    """

    def __init__(
        self,
        *,
        adapter: Any,
        directory: str | os.PathLike[str],
        run_id: str,
        child_name: str,
        trajectory_split_sha256: str,
        run_schema: str,
        resume: Lfm2ClosedLoopRestartCapsuleResume | None = None,
        rotation_keep: int = DEFAULT_RESTART_CAPSULE_ROTATION_KEEP,
        fold_progress_sink: Any | None = None,
        publish_terminal_capsule: bool = False,
    ) -> None:
        self.adapter = adapter
        self.directory = _root_directory(directory)
        self.rotation_keep = _positive_int(rotation_keep, "rotation_keep")
        if not isinstance(publish_terminal_capsule, bool):
            raise TypeError("publish_terminal_capsule must be boolean.")
        self.publish_terminal_capsule = publish_terminal_capsule
        self.adapter_identity = lfm2_restart_capsule_adapter_identity(adapter)
        child_layers = getattr(adapter, "child_layers", None)
        selected = getattr(adapter, "_selected_layers", None)
        if not isinstance(child_layers, tuple) or not child_layers:
            raise Lfm2ClosedLoopRestartCapsuleError(
                "The adapter does not expose its child layer inventory."
            )
        self.bindings = Lfm2ClosedLoopRestartCapsuleBindings(
            run_id=run_id,
            child_name=child_name,
            child_layer_count=len(child_layers),
            selected_layer_indices=tuple(selected or ()),
            adapter_identity_sha256=self.adapter_identity[
                "adapter_identity_sha256"
            ],
            trajectory_split_sha256=trajectory_split_sha256,
            run_schema=run_schema,
        )
        if resume is not None and not isinstance(
            resume, Lfm2ClosedLoopRestartCapsuleResume
        ):
            raise TypeError(
                "resume must be Lfm2ClosedLoopRestartCapsuleResume or None."
            )
        if resume is not None:
            if resume.bindings.bindings_sha256 != self.bindings.bindings_sha256:
                raise Lfm2ClosedLoopRestartCapsuleError(
                    "The resume capsule was sealed for a different run identity."
                )
            if dict(resume.manifest["adapter_identity"]) != dict(
                self.adapter_identity
            ):
                raise Lfm2ClosedLoopRestartCapsuleError(
                    "The resume capsule adapter identity differs from the live "
                    "adapter."
                )
        self.resume = resume
        self._completed: tuple[int, ...] = (
            () if resume is None else resume.completed_layer_indices
        )
        self._sequence = (
            0 if resume is None else resume.reference.capsule_sequence + 1
        )
        self._previous_manifest_sha256 = (
            CLOSED_LOOP_RESTART_CAPSULE_CHAIN_GENESIS_SHA256
            if resume is None
            else resume.reference.manifest_sha256
        )
        self._previous_chain_sha256 = (
            CLOSED_LOOP_RESTART_CAPSULE_CHAIN_GENESIS_SHA256
            if resume is None
            else resume.reference.capsule_chain_sha256
        )
        self._resumed_from_manifest_sha256 = (
            None if resume is None else resume.reference.manifest_sha256
        )
        self._published: list[Lfm2ClosedLoopRestartCapsuleReference] = []
        self._records: list[Mapping[str, Any]] = []
        self._terminal_reference: (
            Lfm2ClosedLoopRestartCapsuleReference | None
        ) = None
        self._fold_progress_sink = fold_progress_sink
        self._progress_chain = (
            None
            if fold_progress_sink is None
            else Lfm2DenseFoldProgressChain(
                fold_progress_sink,
                process_boundary=resume is not None,
            )
        )
        self._first_capsule_of_this_process = True

    # -- orchestrator protocol -----------------------------------------

    def completed_layer_indices(self) -> tuple[int, ...]:
        return self._completed

    def assert_resume_supported(
        self,
        *,
        completed_layer_indices: Sequence[int],
        remaining_layer_indices: Sequence[int],
        selected_layer_indices: Sequence[int],
        body_reclosure_enabled: bool,
        trajectory_split_sha256: str,
        child_state_sha256: str,
        child_topology_sha256: str,
    ) -> Mapping[str, Any]:
        """Admit one mid-body resume or fail closed with a precise reason."""

        completed = _layer_indices(
            completed_layer_indices,
            "completed_layer_indices",
        )
        remaining = _layer_indices(
            remaining_layer_indices,
            "remaining_layer_indices",
        )
        selected = _layer_indices(
            selected_layer_indices,
            "selected_layer_indices",
        )
        if (
            completed != self._completed
            or selected != self.bindings.selected_layer_indices
            or completed + remaining != selected
        ):
            raise Lfm2ClosedLoopRestartCapsuleError(
                "The resume layer inventory differs from the sealed capsule."
            )
        if _digest(
            trajectory_split_sha256,
            "trajectory_split_sha256",
        ) != self.bindings.trajectory_split_sha256:
            raise Lfm2ClosedLoopRestartCapsuleError(
                "The resumed run uses a different deterministic split."
            )
        if self.resume is None:
            raise Lfm2ClosedLoopRestartCapsuleError(
                "A mid-body resume requires a loaded restart capsule."
            )
        fingerprint = getattr(self.adapter, "child_state_fingerprint", None)
        if not callable(fingerprint):
            raise Lfm2ClosedLoopRestartCapsuleError(
                "The adapter does not expose a live child state fingerprint."
            )
        if _digest(child_state_sha256, "child_state_sha256") != fingerprint():
            raise Lfm2ClosedLoopRestartCapsuleError(
                "The live child differs from the child the compiler bound."
            )
        live_state = portable_child_state_sha256(self.adapter.child_model)
        live_topology = portable_child_topology_sha256(self.adapter.child_model)
        if (
            live_state != self.resume.child_state_sha256
            or live_topology != self.resume.child_topology_sha256
        ):
            raise Lfm2ClosedLoopRestartCapsuleError(
                "The live child is not the exact child the capsule restored."
            )
        guard = getattr(self.adapter, "assert_mid_body_resume_supported", None)
        if not callable(guard):
            raise Lfm2ClosedLoopRestartCapsuleError(
                "The adapter cannot certify mid-body resume support."
            )
        adapter_admission = guard(
            completed_layer_indices=completed,
            remaining_layer_indices=remaining,
            body_reclosure_enabled=bool(body_reclosure_enabled),
        )
        if not isinstance(adapter_admission, Mapping):
            raise Lfm2ClosedLoopRestartCapsuleError(
                "The adapter mid-body resume admission is not a mapping."
            )
        body = {
            "schema": CLOSED_LOOP_RESTART_CAPSULE_RESUME_SCHEMA,
            "resumed_capsule_key": self.resume.reference.capsule_key,
            "resumed_capsule_sequence": (
                self.resume.reference.capsule_sequence
            ),
            "resumed_capsule_chain_sha256": (
                self.resume.reference.capsule_chain_sha256
            ),
            "completed_layer_indices": list(completed),
            "remaining_layer_indices": list(remaining),
            "resume_layer_index": remaining[0] if remaining else None,
            "body_reclosure_enabled": bool(body_reclosure_enabled),
            "child_state_sha256": live_state,
            "child_topology_sha256": live_topology,
            "process_local_child_state_sha256": child_state_sha256,
            "process_local_child_topology_sha256": _digest(
                child_topology_sha256,
                "child_topology_sha256",
            ),
            "adapter_admission": dict(adapter_admission),
            "skipped_layers_are_not_replayed": True,
            "rollback_point_is_the_restored_capsule_state": True,
        }
        return {**body, "admission_sha256": canonical_sha256(body)}

    def record_accepted_layer(
        self,
        *,
        layer_index: int,
        layer_ordinal: int,
        completed_layer_indices: Sequence[int],
        remaining_layer_indices: Sequence[int],
        selected_layer_indices: Sequence[int],
        validation_decision_sha256: str,
        child_state_sha256: str,
        child_topology_sha256: str,
        trajectory_split_sha256: str,
    ) -> Mapping[str, Any]:
        """Publish and rotate one sealed capsule for one accepted layer."""

        completed = _layer_indices(
            completed_layer_indices,
            "completed_layer_indices",
        )
        remaining = _layer_indices(
            remaining_layer_indices,
            "remaining_layer_indices",
        )
        selected = _layer_indices(
            selected_layer_indices,
            "selected_layer_indices",
        )
        if (
            selected != self.bindings.selected_layer_indices
            or _digest(trajectory_split_sha256, "trajectory_split_sha256")
            != self.bindings.trajectory_split_sha256
            or completed + remaining != selected
            or layer_index not in completed
        ):
            raise Lfm2ClosedLoopRestartCapsuleError(
                "The accepted-layer inventory differs from the sealed run "
                "identity."
            )
        progress = (
            unavailable_fold_progress_chain(
                process_boundary=(
                    self.resume is not None
                    and self._first_capsule_of_this_process
                ),
            )
            if self._progress_chain is None
            else {
                **self._progress_chain.as_dict(),
                "process_boundary": bool(
                    self.resume is not None
                    and self._first_capsule_of_this_process
                ),
            }
        )
        reference = publish_lfm2_closed_loop_restart_capsule(
            child_model=self.adapter.child_model,
            directory=self.directory,
            bindings=self.bindings,
            adapter_identity=self.adapter_identity,
            layer_index=layer_index,
            layer_ordinal=layer_ordinal,
            completed_layer_indices=completed,
            remaining_layer_indices=remaining,
            validation_decision_sha256=validation_decision_sha256,
            origin_process_local_child_state_sha256=child_state_sha256,
            origin_process_local_child_topology_sha256=child_topology_sha256,
            capsule_sequence=self._sequence,
            previous_capsule_manifest_sha256=self._previous_manifest_sha256,
            previous_capsule_chain_sha256=self._previous_chain_sha256,
            resumed_from_capsule_manifest_sha256=(
                self._resumed_from_manifest_sha256
                if self._first_capsule_of_this_process
                else None
            ),
            fold_progress=progress,
        )
        self._completed = completed
        self._sequence += 1
        self._previous_manifest_sha256 = reference.manifest_sha256
        self._previous_chain_sha256 = reference.capsule_chain_sha256
        self._first_capsule_of_this_process = False
        self._published.append(reference)
        retained = rotate_lfm2_closed_loop_restart_capsules(
            self.directory,
            keep=self.rotation_keep,
        )
        body = {
            "schema": CLOSED_LOOP_RESTART_CAPSULE_RECORD_SCHEMA,
            "layer_index": int(layer_index),
            "layer_ordinal": int(layer_ordinal),
            "capsule": reference.as_dict(),
            "completed_layer_indices": list(completed),
            "remaining_layer_indices": list(remaining),
            "retained_capsule_keys": [
                value.capsule_key for value in retained
            ],
            "rotation_keep": self.rotation_keep,
            "fold_progress": dict(progress),
            "restart_capsule_is_not_a_promotable_child": True,
        }
        record = {**body, "record_sha256": canonical_sha256(body)}
        self._records.append(record)
        return record

    def record_terminal_phase(
        self,
        *,
        completed_layer_indices: Sequence[int],
        selected_layer_indices: Sequence[int],
        validation_decision_sha256: str,
        child_state_sha256: str,
        child_topology_sha256: str,
        trajectory_split_sha256: str,
        body_reclosure_enabled: bool,
        body_reclosure_sha256: str | None,
        final_normalization_sha256: str,
        output_interface_enabled: bool,
        output_interface_sha256: str | None,
    ) -> Mapping[str, Any] | None:
        """Publish one sealed post-output-interface, pre-evaluation capsule.

        Accepted-layer capsules stop at the last body layer, so a crash in the
        evaluators used to cost the whole transaction: the body, the reclosure,
        the final normalization, and the output-interface solves all had to be
        recomputed.  This capsule is the restart point for exactly that window.

        Returns ``None`` when the terminal capsule is not enabled, which the
        orchestrator reads as a declined capsule and records nothing.
        """

        if not self.publish_terminal_capsule:
            return None
        completed = _layer_indices(
            completed_layer_indices,
            "completed_layer_indices",
        )
        selected = _layer_indices(
            selected_layer_indices,
            "selected_layer_indices",
        )
        if (
            selected != self.bindings.selected_layer_indices
            or _digest(trajectory_split_sha256, "trajectory_split_sha256")
            != self.bindings.trajectory_split_sha256
            or completed != selected
        ):
            raise Lfm2ClosedLoopRestartCapsuleError(
                "A terminal capsule requires every selected layer to be "
                "accepted under this exact sealed run identity."
            )
        sealed_phase = lfm2_restart_capsule_terminal_phase(
            body_reclosure_enabled=body_reclosure_enabled,
            body_reclosure_sha256=body_reclosure_sha256,
            final_normalization_sha256=final_normalization_sha256,
            output_interface_enabled=output_interface_enabled,
            output_interface_sha256=output_interface_sha256,
        )
        progress = (
            unavailable_fold_progress_chain(
                process_boundary=(
                    self.resume is not None
                    and self._first_capsule_of_this_process
                ),
            )
            if self._progress_chain is None
            else {
                **self._progress_chain.as_dict(),
                "process_boundary": bool(
                    self.resume is not None
                    and self._first_capsule_of_this_process
                ),
            }
        )
        reference = publish_lfm2_closed_loop_restart_capsule(
            child_model=self.adapter.child_model,
            directory=self.directory,
            bindings=self.bindings,
            adapter_identity=self.adapter_identity,
            layer_index=completed[-1],
            layer_ordinal=len(completed) - 1,
            completed_layer_indices=completed,
            remaining_layer_indices=(),
            validation_decision_sha256=validation_decision_sha256,
            origin_process_local_child_state_sha256=child_state_sha256,
            origin_process_local_child_topology_sha256=child_topology_sha256,
            capsule_sequence=self._sequence,
            previous_capsule_manifest_sha256=self._previous_manifest_sha256,
            previous_capsule_chain_sha256=self._previous_chain_sha256,
            resumed_from_capsule_manifest_sha256=(
                self._resumed_from_manifest_sha256
                if self._first_capsule_of_this_process
                else None
            ),
            fold_progress=progress,
            capsule_phase=_TERMINAL_PHASE,
            terminal_phase=sealed_phase,
        )
        self._completed = completed
        self._sequence += 1
        self._previous_manifest_sha256 = reference.manifest_sha256
        self._previous_chain_sha256 = reference.capsule_chain_sha256
        self._first_capsule_of_this_process = False
        self._published.append(reference)
        self._terminal_reference = reference
        retained = rotate_lfm2_closed_loop_restart_capsules(
            self.directory,
            keep=self.rotation_keep,
        )
        body = {
            "schema": CLOSED_LOOP_RESTART_CAPSULE_RECORD_SCHEMA,
            "capsule_phase": _TERMINAL_PHASE,
            "layer_index": int(completed[-1]),
            "layer_ordinal": int(len(completed) - 1),
            "capsule": reference.as_dict(),
            "completed_layer_indices": list(completed),
            "remaining_layer_indices": [],
            "retained_capsule_keys": [
                value.capsule_key for value in retained
            ],
            "rotation_keep": self.rotation_keep,
            "fold_progress": dict(progress),
            "terminal_phase": dict(sealed_phase),
            "restart_capsule_is_not_a_promotable_child": True,
        }
        record = {**body, "record_sha256": canonical_sha256(body)}
        self._records.append(record)
        return record

    def as_dict(self) -> Mapping[str, Any]:
        body = {
            "schema": CLOSED_LOOP_RESTART_CAPSULE_WRITER_SCHEMA,
            "directory": str(self.directory),
            "rotation_keep": self.rotation_keep,
            "bindings": self.bindings.as_dict(),
            "adapter_identity": dict(self.adapter_identity),
            "resumed": self.resume is not None,
            "resumed_from": (
                None if self.resume is None else self.resume.as_dict()
            ),
            "published_capsule_keys": [
                value.capsule_key for value in self._published
            ],
            "published_capsule_count": len(self._published),
            "next_capsule_sequence": self._sequence,
            "fold_progress_chain_bound": self._progress_chain is not None,
            "restart_capsule_is_not_a_promotable_child": True,
        }
        if self._terminal_reference is not None:
            # A writer that never published a terminal capsule keeps its
            # historical sealed report body.
            body = {
                **body,
                "terminal_capsule_published": True,
                "terminal_capsule_key": self._terminal_reference.capsule_key,
                "terminal_capsule_chain_sha256": (
                    self._terminal_reference.capsule_chain_sha256
                ),
            }
        return {**body, "writer_sha256": canonical_sha256(body)}


__all__ = [
    "CLOSED_LOOP_RESTART_CAPSULE_BINDINGS_SCHEMA",
    "CLOSED_LOOP_RESTART_CAPSULE_CHAIN_GENESIS_SHA256",
    "CLOSED_LOOP_RESTART_CAPSULE_CHAIN_SCHEMA",
    "CLOSED_LOOP_RESTART_CAPSULE_CHILD_STATE_SCHEMA",
    "CLOSED_LOOP_RESTART_CAPSULE_KEY_SCHEMA",
    "CLOSED_LOOP_RESTART_CAPSULE_PROGRESS_SCHEMA",
    "CLOSED_LOOP_RESTART_CAPSULE_RECORD_SCHEMA",
    "CLOSED_LOOP_RESTART_CAPSULE_RESUME_SCHEMA",
    "CLOSED_LOOP_RESTART_CAPSULE_SCHEMA",
    "CLOSED_LOOP_RESTART_CAPSULE_TERMINAL_PHASE_SCHEMA",
    "CLOSED_LOOP_RESTART_CAPSULE_TOPOLOGY_SCHEMA",
    "CLOSED_LOOP_RESTART_CAPSULE_WRITER_SCHEMA",
    "DEFAULT_RESTART_CAPSULE_ROTATION_KEEP",
    "Lfm2ClosedLoopRestartCapsuleAuthentication",
    "Lfm2ClosedLoopRestartCapsuleBindings",
    "Lfm2ClosedLoopRestartCapsuleError",
    "Lfm2ClosedLoopRestartCapsuleReference",
    "Lfm2ClosedLoopRestartCapsuleResume",
    "Lfm2ClosedLoopRestartCapsuleWriter",
    "Lfm2DenseFoldProgressChain",
    "authenticate_lfm2_closed_loop_restart_capsule",
    "child_state_alias_groups",
    "discover_lfm2_closed_loop_restart_capsules",
    "latest_lfm2_closed_loop_restart_capsule",
    "lfm2_restart_capsule_adapter_identity",
    "lfm2_restart_capsule_split_sha256",
    "lfm2_restart_capsule_terminal_phase",
    "load_lfm2_closed_loop_restart_capsule",
    "portable_child_state_sha256",
    "portable_child_topology_sha256",
    "publish_lfm2_closed_loop_restart_capsule",
    "rotate_lfm2_closed_loop_restart_capsules",
    "unavailable_fold_progress_chain",
]
