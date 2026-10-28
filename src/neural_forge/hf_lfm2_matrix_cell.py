"""Authenticated component-by-technique matrix cells for LFM2 folding.

This module is orchestration, not another fold compiler.  Technique providers
produce one candidate for one component/site/level.  The runner below makes
their evidence comparable, keeps numerical triage separate from the manual
intelligence verdict, and records enough provenance to resume without
repeating invariant evidence preparation.

Two evidence classes are explicit:

``isolation_only``
    One component is replaced in an otherwise exact parent shell.  This is
    fast fault-localisation evidence and never proves whole-model compaction.

``complete_child``
    The same technique is installed into a separately materialised blank
    target while every non-focused target path follows the selected baseline.
    Only this class may support a parameter-compaction claim.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import inspect
import json
import math
import os
import re
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol

from .dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from .dynamic_fold_phase_cache import FoldPhaseKey


REQUEST_SCHEMA = "neural-forge-lfm2-matrix-cell-request-v1"
RECEIPT_SCHEMA = "neural-forge-lfm2-matrix-cell-receipt-v2"
PROGRESS_SCHEMA = "neural-forge-lfm2-matrix-cell-progress-event-v1"
LEDGER_SCHEMA = "neural-forge-lfm2-matrix-ledger-record-v1"
EXECUTOR_SCHEMA = "neural-forge-lfm2-matrix-technique-executor-v1"
FAILURE_SCHEMA = "neural-forge-lfm2-matrix-cell-failure-v1"

EvidenceClass = Literal["isolation_only", "complete_child"]
ExecutionStatus = Literal["complete", "timeout", "infrastructure_failure"]

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+-]*\Z")
_SITE = re.compile(r"(?:GLOBAL|FINAL|L(?:0[0-9]|1[0-3]))\Z")
_CELL = re.compile(
    r"(?:(?:C(?:0[1-9]|1[0-5])(?:\+C(?:0[1-9]|1[0-5]))*)\."
    r"T[1-9]\.(?:M16|M8|M4|X4|NA)\.(?:GLOBAL|FINAL|L(?:0[0-9]|1[0-3]))|"
    r"CTRL\.copy_exact|CAL\.C01\.T5\.X4)\Z"
)

_FORBIDDEN_FOLD_OPERATIONS = (
    "autograd_grad_calls",
    "backward_calls",
    "optimizer_constructions",
    "optimizer_steps",
    "training_steps",
    "pruning_operations",
    "quantization_operations",
    "parent_trace_recaptures",
    "network_requests",
)

_NUMERIC_ROLE_KEYS = frozenset(
    {
        "source_storage",
        "source_forward",
        "child_forward",
        "accumulation",
        "factorization",
        "target_publication",
        "runtime_verification",
    }
)

_REQUIRED_BACKEND_STAGES = (
    "evidence_opened",
    "checkpoint_reuse_resolved",
    "copy_exact_control_materialized",
    "calibration_completed",
    "component_solve_completed",
    "component_installed",
    "boundary_evaluation_completed",
    "composed_evaluation_completed",
    "free_run_completed",
    "decode_smoke_completed",
)

_REQUIRED_CACHE_PHASES = (
    "evidence_preparation",
    "target_materialization",
    "technique_solve",
    "component_install",
    "behavior_evaluation",
)

_COMPONENT_METRIC_GROUPS = {
    "weight",
    "component_write",
    "macrocell",
    "terminal",
    "attention_relation",
    "mlp_relation",
    "state_mixer_relation",
    "interface_relation",
    "cache_trajectory",
}

_COMPONENTS = {
    "C01": "token_interface_in",
    "C02": "token_interface_out",
    "C03": "conv_in_proj",
    "C04": "conv_kernel",
    "C05": "conv_out_proj",
    "C06": "attention_query",
    "C07": "attention_key",
    "C08": "attention_value",
    "C09": "attention_output",
    "C10": "attention_per_head",
    "C11": "mlp_gate",
    "C12": "mlp_up",
    "C13": "mlp_down",
    "C14": "normalization",
    "C15": "lm_head",
}

_TECHNIQUES = {
    "T1": "dense_prior_anchored_ridge",
    "T2": "cross_width_residual_chart",
    "T3": "balanced_controllability_observability",
    "T4": "low_rank_factor_writer",
    "T5": "sparse_residual_omp_routes",
    "T6": "kind_translation",
    "T7": "attention_relation_matching",
    "T8": "operator_gram_relation_matching",
    "T9": "progressive_multi_pass",
}

_LEVELS = {
    "M16": 1.0 / 16.0,
    "M8": 1.0 / 8.0,
    "M4": 1.0 / 4.0,
    "X4": None,
    "NA": 0.0,
}

_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, threading.RLock] = {}


class Lfm2MatrixCellError(RuntimeError):
    """A request, execution, receipt, or ledger failed authentication."""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2MatrixCellError(f"{label} must be nonempty text.")
    return value.strip()


def _identifier(value: Any, label: str) -> str:
    result = _text(value, label)
    if _IDENTIFIER.fullmatch(result) is None:
        raise Lfm2MatrixCellError(f"{label} is not a canonical identifier.")
    return result


def _digest(value: Any, label: str) -> str:
    result = _text(value, label)
    if _DIGEST.fullmatch(result) is None:
        raise Lfm2MatrixCellError(f"{label} must be a lowercase SHA-256 digest.")
    return result


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise Lfm2MatrixCellError(f"{label} must be a {qualifier} integer.")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Lfm2MatrixCellError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise Lfm2MatrixCellError(f"{label} must be finite.")
    return result


def _plain_json(value: Any, label: str) -> Any:
    """Validate finite JSON data and return a detached plain representation."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Lfm2MatrixCellError(f"{label} contains a non-finite number.")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key = _text(key, f"{label} key")
            if key in result:
                raise Lfm2MatrixCellError(f"{label} repeats key {key!r}.")
            result[key] = _plain_json(item, f"{label}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _plain_json(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    raise Lfm2MatrixCellError(f"{label} is not finite JSON data.")


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        missing = sorted(expected - set(value if isinstance(value, Mapping) else ()))
        unknown = sorted(set(value if isinstance(value, Mapping) else ()) - expected)
        raise Lfm2MatrixCellError(
            f"{label} fields differ: missing={missing}, unknown={unknown}."
        )
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _path_lock(path: Path) -> threading.RLock:
    key = os.path.normcase(str(path.expanduser().absolute()))
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _exclusive_file(path: Path):
    """Cross-process advisory lock for one append-only artifact."""

    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _path_lock(lock_path), lock_path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True, slots=True)
class MatrixCellRequest:
    instance_id: str
    cell_id: str
    component_ids: tuple[str, ...]
    technique_id: str
    level_id: str
    site_id: str
    evidence_class: EvidenceClass
    campaign_wave: int
    parent_parameter_count: int
    target_parameter_count: int | None
    evidence_binding: Mapping[str, Any]
    execution_policy: Mapping[str, Any]
    technique_configuration: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "instance_id", _identifier(self.instance_id, "instance_id"))
        cell_id = _text(self.cell_id, "cell_id")
        if _CELL.fullmatch(cell_id) is None:
            raise Lfm2MatrixCellError("cell_id is not a canonical campaign cell.")
        object.__setattr__(self, "cell_id", cell_id)
        components = tuple(_text(value, "component_id") for value in self.component_ids)
        if (
            not components
            or len(set(components)) != len(components)
            or any(value not in _COMPONENTS for value in components)
        ):
            raise Lfm2MatrixCellError("component_ids are empty, repeated, or unknown.")
        object.__setattr__(self, "component_ids", components)
        if self.technique_id not in _TECHNIQUES:
            raise Lfm2MatrixCellError("technique_id is unknown.")
        if self.level_id not in _LEVELS:
            raise Lfm2MatrixCellError("level_id is unknown.")
        if _SITE.fullmatch(self.site_id) is None:
            raise Lfm2MatrixCellError("site_id is invalid.")
        if self.evidence_class not in {"isolation_only", "complete_child"}:
            raise Lfm2MatrixCellError("evidence_class is invalid.")
        _positive_int(self.campaign_wave, "campaign_wave", allow_zero=True)
        _positive_int(self.parent_parameter_count, "parent_parameter_count")
        if self.target_parameter_count is not None:
            _positive_int(self.target_parameter_count, "target_parameter_count")
        if self.evidence_class == "complete_child":
            if (
                self.target_parameter_count is None
                or self.target_parameter_count >= self.parent_parameter_count
            ):
                raise Lfm2MatrixCellError(
                    "complete_child evidence requires a smaller exact target."
                )
        object.__setattr__(
            self,
            "evidence_binding",
            MappingProxyType(_plain_json(self.evidence_binding, "evidence_binding")),
        )
        object.__setattr__(
            self,
            "execution_policy",
            MappingProxyType(_plain_json(self.execution_policy, "execution_policy")),
        )
        object.__setattr__(
            self,
            "technique_configuration",
            MappingProxyType(
                _plain_json(self.technique_configuration, "technique_configuration")
            ),
        )
        if self.evidence_binding.get("evidence_sha256") is None:
            raise Lfm2MatrixCellError("evidence_binding has no evidence_sha256.")
        _digest(self.evidence_binding["evidence_sha256"], "evidence_sha256")
        if self.cell_id not in {"CTRL.copy_exact", "CAL.C01.T5.X4"}:
            expected_prefix = "+".join(self.component_ids)
            expected = (
                f"{expected_prefix}.{self.technique_id}.{self.level_id}.{self.site_id}"
            )
            if self.cell_id != expected:
                raise Lfm2MatrixCellError("cell_id differs from its explicit axes.")

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict(include_fingerprint=False))

    @property
    def component_names(self) -> tuple[str, ...]:
        return tuple(_COMPONENTS[value] for value in self.component_ids)

    @property
    def technique_name(self) -> str:
        return _TECHNIQUES[self.technique_id]

    @property
    def compaction_claim_permitted(self) -> bool:
        return self.evidence_class == "complete_child"

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        body = {
            "schema": REQUEST_SCHEMA,
            "instance_id": self.instance_id,
            "cell_id": self.cell_id,
            "component_ids": list(self.component_ids),
            "component_names": list(self.component_names),
            "technique_id": self.technique_id,
            "technique_name": self.technique_name,
            "level_id": self.level_id,
            "nominal_component_fraction_removed": _LEVELS[self.level_id],
            "site_id": self.site_id,
            "evidence_class": self.evidence_class,
            "campaign_wave": self.campaign_wave,
            "parent_parameter_count": self.parent_parameter_count,
            "target_parameter_count": self.target_parameter_count,
            "evidence_binding": dict(self.evidence_binding),
            "execution_policy": dict(self.execution_policy),
            "technique_configuration": dict(self.technique_configuration),
            "compaction_claim_permitted": self.compaction_claim_permitted,
        }
        if include_fingerprint:
            body["request_sha256"] = canonical_sha256(body)
        return body

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MatrixCellRequest:
        row = _exact_keys(
            value,
            {
                "schema",
                "instance_id",
                "cell_id",
                "component_ids",
                "component_names",
                "technique_id",
                "technique_name",
                "level_id",
                "nominal_component_fraction_removed",
                "site_id",
                "evidence_class",
                "campaign_wave",
                "parent_parameter_count",
                "target_parameter_count",
                "evidence_binding",
                "execution_policy",
                "technique_configuration",
                "compaction_claim_permitted",
                "request_sha256",
            },
            "matrix cell request",
        )
        if row["schema"] != REQUEST_SCHEMA:
            raise Lfm2MatrixCellError("Matrix request schema is unsupported.")
        result = cls(
            instance_id=row["instance_id"],
            cell_id=row["cell_id"],
            component_ids=tuple(row["component_ids"]),
            technique_id=row["technique_id"],
            level_id=row["level_id"],
            site_id=row["site_id"],
            evidence_class=row["evidence_class"],
            campaign_wave=row["campaign_wave"],
            parent_parameter_count=row["parent_parameter_count"],
            target_parameter_count=row["target_parameter_count"],
            evidence_binding=row["evidence_binding"],
            execution_policy=row["execution_policy"],
            technique_configuration=row["technique_configuration"],
        )
        if result.as_dict() != dict(row):
            raise Lfm2MatrixCellError("Matrix request is stale or noncanonical.")
        return result


class MatrixProgressLog:
    """Durable hash-chained cell progress with monotonic sequence numbers."""

    def __init__(self, path: str | os.PathLike[str], request_sha256: str) -> None:
        self.path = Path(path).expanduser().absolute()
        self.request_sha256 = _digest(request_sha256, "request_sha256")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._events = self._read()

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events = []
        previous: str | None = None
        for index, line in enumerate(self.path.read_text(encoding="utf-8").splitlines()):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Lfm2MatrixCellError("Progress JSONL is malformed.") from exc
            event_sha = row.pop("event_sha256", None)
            if (
                row.get("schema") != PROGRESS_SCHEMA
                or row.get("request_sha256") != self.request_sha256
                or row.get("sequence") != index
                or row.get("previous_event_sha256") != previous
                or canonical_sha256(row) != event_sha
            ):
                raise Lfm2MatrixCellError("Progress chain authentication failed.")
            row["event_sha256"] = event_sha
            events.append(row)
            previous = event_sha
        return events

    @property
    def events(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(MappingProxyType(dict(value)) for value in self._events)

    @property
    def stages(self) -> tuple[str, ...]:
        return tuple(str(value["stage"]) for value in self._events)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self._events)

    def emit(self, stage: str, **data: Any) -> Mapping[str, Any]:
        stage = _identifier(stage, "progress stage")
        values = _plain_json(data, "progress data")
        with _exclusive_file(self.path):
            self._events = self._read()
            body = {
                "schema": PROGRESS_SCHEMA,
                "request_sha256": self.request_sha256,
                "sequence": len(self._events),
                "previous_event_sha256": (
                    None if not self._events else self._events[-1]["event_sha256"]
                ),
                "recorded_at_utc": _now(),
                "stage": stage,
                "data": values,
            }
            row = {**body, "event_sha256": canonical_sha256(body)}
            with self.path.open("ab") as handle:
                handle.write(canonical_json_bytes(row))
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._events.append(row)
        return MappingProxyType(row)


class MatrixTechniqueExecute(Protocol):
    def __call__(
        self,
        request: MatrixCellRequest,
        run_directory: Path,
        emit: Callable[..., Mapping[str, Any]],
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class MatrixTechniqueExecutor:
    technique_id: str
    implementation_id: str
    implementation_sha256: str
    execute: MatrixTechniqueExecute

    def __post_init__(self) -> None:
        if self.technique_id not in _TECHNIQUES:
            raise Lfm2MatrixCellError("Executor technique_id is unknown.")
        object.__setattr__(
            self,
            "implementation_id",
            _identifier(self.implementation_id, "implementation_id"),
        )
        object.__setattr__(
            self,
            "implementation_sha256",
            _digest(self.implementation_sha256, "implementation_sha256"),
        )
        if not callable(self.execute):
            raise Lfm2MatrixCellError("Executor execute must be callable.")

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": EXECUTOR_SCHEMA,
            "technique_id": self.technique_id,
            "technique_name": _TECHNIQUES[self.technique_id],
            "implementation_id": self.implementation_id,
            "implementation_sha256": self.implementation_sha256,
        }


def build_matrix_phase_key(
    request: MatrixCellRequest,
    executor: MatrixTechniqueExecutor,
    *,
    phase_name: str,
    numeric_contract: Mapping[str, Any],
    dependency_sha256s: Mapping[str, str] | None = None,
    abi: Mapping[str, Any] | None = None,
    request_projection: Mapping[str, Any] | None = None,
) -> FoldPhaseKey:
    """Bind reusable matrix math to every input that can change its answer.

    The default seals the complete request.  A caller may instead provide a
    phase-specific, input-only projection so invariant work (for example one
    site evidence extraction) is shared across technique cells.  The
    projection is itself authenticated and may not be empty; callers must put
    every answer-changing field either there, in ``dependency_sha256s``, in the
    numeric contract, or in the ABI.
    """

    if not isinstance(request, MatrixCellRequest) or not isinstance(
        executor, MatrixTechniqueExecutor
    ):
        raise Lfm2MatrixCellError("Matrix phase keys need a request and executor.")
    if request_projection is None:
        projection = request.as_dict(include_fingerprint=False)
        projection_kind = "complete_request"
    else:
        projection = _plain_json(request_projection, "phase request projection")
        if not projection:
            raise Lfm2MatrixCellError("Matrix phase request projection is empty.")
        projection_kind = "phase_specific"
    dependencies = {
        "request": canonical_sha256(projection),
        "evidence": _digest(
            request.evidence_binding["evidence_sha256"], "evidence_sha256"
        ),
        "executor": executor.fingerprint,
    }
    for name, digest in (dependency_sha256s or {}).items():
        identifier = _identifier(name, "phase dependency name")
        if identifier in dependencies:
            raise Lfm2MatrixCellError(
                f"Matrix phase dependency {identifier!r} is reserved."
            )
        dependencies[identifier] = _digest(digest, f"phase dependency {identifier}")
    canonical_abi = {
        "matrix_request_schema": REQUEST_SCHEMA,
        "matrix_executor_schema": EXECUTOR_SCHEMA,
        "technique_id": request.technique_id,
        "component_ids": list(request.component_ids),
        "level_id": request.level_id,
        "site_id": request.site_id,
        "evidence_class": request.evidence_class,
        "request_projection_kind": projection_kind,
        "request_projection_sha256": dependencies["request"],
    }
    extra_abi = _plain_json({} if abi is None else abi, "phase abi")
    reserved_abi = sorted(set(canonical_abi).intersection(extra_abi))
    if reserved_abi:
        raise Lfm2MatrixCellError(
            "Matrix phase ABI cannot override canonical fields: "
            + ", ".join(reserved_abi)
        )
    phase_abi = {**canonical_abi, **extra_abi}
    return FoldPhaseKey(
        phase_name=_identifier(phase_name, "phase_name"),
        dependency_sha256s=dependencies,
        numeric_contract=_plain_json(numeric_contract, "numeric_contract"),
        abi=phase_abi,
    )


def source_bound_executor(
    technique_id: str,
    implementation_id: str,
    execute: MatrixTechniqueExecute,
) -> MatrixTechniqueExecutor:
    """Bind an executor to the source bytes of its implementation callable."""

    try:
        source = inspect.getsource(execute).encode("utf-8")
    except (OSError, TypeError) as exc:
        raise Lfm2MatrixCellError(
            "Technique executor source must be inspectable for authentication."
        ) from exc
    return MatrixTechniqueExecutor(
        technique_id=technique_id,
        implementation_id=implementation_id,
        implementation_sha256=hashlib.sha256(source).hexdigest(),
        execute=execute,
    )


def load_executor(reference: str) -> MatrixTechniqueExecutor:
    """Load ``module:attribute`` and require a sealed executor or factory."""

    module_name, separator, attribute = _text(reference, "executor reference").partition(":")
    if not separator or not module_name or not attribute:
        raise Lfm2MatrixCellError("Executor reference must be module:attribute.")
    value = getattr(importlib.import_module(module_name), attribute)
    executor = value() if callable(value) and not isinstance(value, MatrixTechniqueExecutor) else value
    if not isinstance(executor, MatrixTechniqueExecutor):
        raise Lfm2MatrixCellError("Executor reference did not resolve to an executor.")
    return executor


def _validate_operation_audit(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2MatrixCellError("operation_audit must be an object.")
    audit = _plain_json(value, "operation_audit")
    missing = sorted(set(_FORBIDDEN_FOLD_OPERATIONS) - set(audit))
    if missing:
        raise Lfm2MatrixCellError(f"operation_audit is missing {missing}.")
    violations = {
        name: audit[name]
        for name in _FORBIDDEN_FOLD_OPERATIONS
        if audit[name] not in {0, False}
    }
    if violations:
        raise Lfm2MatrixCellError(
            "Technique execution violated fold-only provenance: "
            + ", ".join(f"{key}={value}" for key, value in violations.items())
        )
    return audit


def _validate_numeric_execution(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2MatrixCellError("numeric_execution must be an object.")
    if set(value) != _NUMERIC_ROLE_KEYS:
        raise Lfm2MatrixCellError(
            "numeric_execution must resolve every independent numeric role."
        )
    result = _plain_json(value, "numeric_execution")
    for role, row in result.items():
        if not isinstance(row, Mapping) or not {"dtype", "device"}.issubset(row):
            raise Lfm2MatrixCellError(
                f"numeric_execution.{role} lacks dtype/device policy."
            )
    return result


def _validate_behavior(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2MatrixCellError("behavior must be an object.")
    behavior = _plain_json(value, "behavior")
    free_run = behavior.get("free_run")
    smoke = behavior.get("decode_smoke")
    if not isinstance(free_run, Mapping) or not isinstance(smoke, list) or not smoke:
        raise Lfm2MatrixCellError("behavior lacks free_run or decode_smoke evidence.")
    _finite_number(free_run.get("token_match_vs_control"), "free-run token match")
    _finite_number(free_run.get("mean_lcp"), "free-run mean_lcp")
    for index, row in enumerate(smoke):
        if not isinstance(row, Mapping):
            raise Lfm2MatrixCellError("decode_smoke rows must be objects.")
        termination = row.get("termination_reason")
        if termination not in {"eos", "cap", "loop", "error"}:
            raise Lfm2MatrixCellError(
                f"decode_smoke[{index}] has an invalid termination_reason."
            )
        if not isinstance(row.get("output_verbatim"), str):
            raise Lfm2MatrixCellError("decode_smoke output must be verbatim text.")
        if "safety_observation" not in row:
            raise Lfm2MatrixCellError(
                "decode_smoke must keep safety observations on a separate axis."
            )
    return behavior


def _validate_checkpoint_reuse(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2MatrixCellError("checkpoint_reuse must be an object.")
    result = _plain_json(value, "checkpoint_reuse")
    missing = sorted(set(_REQUIRED_CACHE_PHASES) - set(result))
    if missing:
        raise Lfm2MatrixCellError(
            f"checkpoint_reuse is missing reusable phases {missing}."
        )
    for phase in _REQUIRED_CACHE_PHASES:
        row = result[phase]
        if not isinstance(row, Mapping):
            raise Lfm2MatrixCellError(f"checkpoint_reuse.{phase} must be an object.")
        if row.get("status") not in {"hit", "miss", "not_applicable"}:
            raise Lfm2MatrixCellError(
                f"checkpoint_reuse.{phase}.status is invalid."
            )
        if row.get("status") == "not_applicable":
            if row.get("cache_key_sha256") is not None:
                raise Lfm2MatrixCellError(
                    f"checkpoint_reuse.{phase} has a key but is not applicable."
                )
        else:
            _digest(
                row.get("cache_key_sha256"),
                f"checkpoint_reuse.{phase}.cache_key_sha256",
            )
        _text(row.get("reason"), f"checkpoint_reuse.{phase}.reason")
    return result


def _validate_phase_timings(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2MatrixCellError("phase_timings must be an object.")
    result = _plain_json(value, "phase_timings")
    missing = sorted(set(_REQUIRED_BACKEND_STAGES) - set(result))
    if missing:
        raise Lfm2MatrixCellError(f"phase_timings is missing stages {missing}.")
    for stage in _REQUIRED_BACKEND_STAGES:
        row = result[stage]
        if not isinstance(row, Mapping):
            raise Lfm2MatrixCellError(f"phase_timings.{stage} must be an object.")
        elapsed = _finite_number(
            row.get("wall_seconds"), f"phase_timings.{stage}.wall_seconds"
        )
        if elapsed < 0:
            raise Lfm2MatrixCellError("Phase wall time cannot be negative.")
        for resource in ("peak_rss_bytes", "peak_vram_bytes"):
            _positive_int(
                row.get(resource),
                f"phase_timings.{stage}.{resource}",
                allow_zero=True,
            )
        if row.get("cache_status") not in {
            "hit",
            "miss",
            "not_applicable",
            "mixed",
        }:
            raise Lfm2MatrixCellError(
                f"phase_timings.{stage}.cache_status is invalid."
            )
    return result


def _validate_component_diagnostics(value: Any) -> dict[str, Any]:
    row = _exact_keys(
        value,
        {
            "measurement_component_ids",
            "physical_component_ids",
            "metric_groups",
            "numeric_decision_authority",
        },
        "component_diagnostics",
    )
    result = _plain_json(row, "component_diagnostics")
    for key in ("measurement_component_ids", "physical_component_ids"):
        values = result[key]
        if not isinstance(values, list) or not values:
            raise Lfm2MatrixCellError(f"component_diagnostics.{key} is empty.")
        for value in values:
            _identifier(value, f"component_diagnostics.{key}")
    groups = result["metric_groups"]
    if not isinstance(groups, Mapping) or set(groups) != _COMPONENT_METRIC_GROUPS:
        raise Lfm2MatrixCellError(
            "component_diagnostics.metric_groups must resolve every diagnostic axis."
        )
    if any(value is not None and not isinstance(value, Mapping) for value in groups.values()):
        raise Lfm2MatrixCellError("Component metric groups must be objects or null.")
    if result["numeric_decision_authority"] is not False:
        raise Lfm2MatrixCellError(
            "Component numerical diagnostics cannot decide intelligence."
        )
    return result


def _diagnostic_first_divergence(result: Mapping[str, Any]) -> str | None:
    """Return metric triage only; this never accepts or rejects intelligence."""

    calibration = result["calibration"]
    solve = result["solve"]
    boundary = result["boundary"]
    composed = result["composed"]
    behavior = result["behavior"]
    if calibration.get("passed") is not True:
        return "calibration"
    requested = solve.get("requested_clamp_ratio")
    allowed = solve.get("allowed_clamp_ratio")
    if (
        solve.get("condition_cap_saturated") is True
        or (
            solve.get("clamp_engaged") is True
            and isinstance(requested, (int, float))
            and isinstance(allowed, (int, float))
            and allowed > 0
            and requested / allowed > 4.0
        )
    ):
        return "solve_conditioning"
    if _finite_number(
        boundary.get("write_heldout_cos"), "boundary write cosine"
    ) < 0.95:
        return "local_solve"
    if _finite_number(
        boundary.get("post_operator_heldout_rel_l2"),
        "post-operator relative L2",
    ) > 0.4:
        return "macrocell_band"
    if _finite_number(
        composed.get("vs_copy_exact_control_delta"),
        "copy-exact control delta",
    ) > 0.02:
        return "composition"
    if _finite_number(
        behavior["free_run"].get("token_match_vs_control"),
        "free-run token match",
    ) < 0.90:
        return "free_run"
    return None


def _validate_backend_result(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "code_identity",
        "evidence",
        "calibration",
        "solve",
        "boundary",
        "composed",
        "behavior",
        "operation_audit",
        "numeric_execution",
        "resources",
        "checkpoint_reuse",
        "phase_timings",
        "component_diagnostics",
        "artifacts",
    }
    row = _exact_keys(value, expected, "technique result")
    result = _plain_json(row, "technique result")
    for name in (
        "code_identity",
        "evidence",
        "calibration",
        "solve",
        "boundary",
        "composed",
        "resources",
        "checkpoint_reuse",
        "phase_timings",
        "component_diagnostics",
        "artifacts",
    ):
        if not isinstance(result[name], Mapping):
            raise Lfm2MatrixCellError(f"{name} must be an object.")
    result["operation_audit"] = _validate_operation_audit(
        result["operation_audit"]
    )
    result["numeric_execution"] = _validate_numeric_execution(
        result["numeric_execution"]
    )
    result["behavior"] = _validate_behavior(result["behavior"])
    result["checkpoint_reuse"] = _validate_checkpoint_reuse(
        result["checkpoint_reuse"]
    )
    result["phase_timings"] = _validate_phase_timings(result["phase_timings"])
    result["component_diagnostics"] = _validate_component_diagnostics(
        result["component_diagnostics"]
    )
    for container, key in (
        (result["boundary"], "write_heldout_cos"),
        (result["boundary"], "write_heldout_rel_l2"),
        (result["boundary"], "post_operator_heldout_cos"),
        (result["boundary"], "post_operator_heldout_rel_l2"),
        (result["composed"], "terminal_heldout_rel_l2"),
        (result["composed"], "vs_copy_exact_control_delta"),
    ):
        _finite_number(container.get(key), key)
    if result["calibration"].get("passed") not in {True, False}:
        raise Lfm2MatrixCellError("calibration.passed must be boolean.")
    return result


def _validate_evidence_class_result(
    request: MatrixCellRequest,
    result: Mapping[str, Any],
) -> None:
    """Prevent a parent-shell diagnostic from masquerading as compaction."""

    artifacts = result["artifacts"]
    diagnostics = result["component_diagnostics"]
    if tuple(diagnostics["measurement_component_ids"]) != request.component_ids:
        raise Lfm2MatrixCellError(
            "Component diagnostics differ from the request measurement roles."
        )
    if request.evidence_class == "isolation_only":
        if (
            artifacts.get("execution_surface")
            != "parent_shell_component_substitution"
            or artifacts.get("physical_child_materialized") is not False
            or artifacts.get("stored_parameter_count_claim_authority") is not False
        ):
            raise Lfm2MatrixCellError(
                "Isolation evidence must identify a parent-shell substitution "
                "and disclaim physical compaction authority."
            )
        return
    required = {
        "execution_surface": "blank_exact_budget_child",
        "physical_child_materialized": True,
        "all_target_paths_owned_once": True,
        "nonfocus_paths_baseline_byte_identical": True,
        "save_reload_state_identical": True,
    }
    mismatched = {
        key: artifacts.get(key)
        for key, expected in required.items()
        if artifacts.get(key) != expected
    }
    if mismatched:
        raise Lfm2MatrixCellError(
            "Complete-child physical proof is missing or false: "
            + ", ".join(f"{key}={value!r}" for key, value in mismatched.items())
        )
    if (
        artifacts.get("physical_parent_parameter_count")
        != request.parent_parameter_count
        or artifacts.get("physical_target_parameter_count")
        != request.target_parameter_count
    ):
        raise Lfm2MatrixCellError(
            "Complete-child physical counts differ from the sealed request."
        )
    for key in (
        "physical_parameter_manifest_sha256",
        "serialized_child_sha256",
        "save_reload_state_sha256",
    ):
        _digest(artifacts.get(key), f"artifacts.{key}")


def authenticate_receipt(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2MatrixCellError("Receipt must be an object.")
    receipt_sha = value.get("receipt_sha256")
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if body.get("schema") != RECEIPT_SCHEMA or canonical_sha256(body) != receipt_sha:
        raise Lfm2MatrixCellError("Receipt seal authentication failed.")
    request = MatrixCellRequest.from_dict(body["request"])
    if body.get("request_sha256") != request.fingerprint:
        raise Lfm2MatrixCellError("Receipt request binding is invalid.")
    if body.get("automatic_intelligence_verdict_performed") is not False:
        raise Lfm2MatrixCellError("Receipt falsely claims an automatic verdict.")
    if body.get("manual_intelligence_review", {}).get("status") not in {
        "pending",
        "useful",
        "partial",
        "incoherent",
        "censored",
    }:
        raise Lfm2MatrixCellError("Receipt manual-review state is invalid.")
    return MappingProxyType(_plain_json(value, "receipt"))


def record_manual_intelligence_review(
    receipt_path: str | os.PathLike[str],
    *,
    status: Literal["useful", "partial", "incoherent", "censored"],
    reviewer_kind: Literal["human", "agent_manual"],
    summary: str,
    prompt_reviews: Sequence[Mapping[str, Any]],
    ledger_path: str | os.PathLike[str] | None = None,
) -> Mapping[str, Any]:
    """Attach a non-automatic whole-response review as a sealed revision.

    No score is derived here.  The caller must read the verbatim outputs and
    supply judgments.  Safety rows are explicitly not intelligence-scored and
    output caps are recorded as censored endpoints, never automatic failures.
    """

    path = Path(receipt_path).expanduser().absolute()
    current = authenticate_receipt(json.loads(path.read_text(encoding="utf-8")))
    rows = []
    for index, value in enumerate(prompt_reviews):
        row = _exact_keys(
            value,
            {
                "prompt_id",
                "is_safety",
                "intelligence_assessment",
                "termination_reason",
                "termination_censored",
                "rationale",
            },
            f"prompt_reviews[{index}]",
        )
        prompt_id = _identifier(row["prompt_id"], f"prompt_reviews[{index}].prompt_id")
        if not isinstance(row["is_safety"], bool):
            raise Lfm2MatrixCellError("Manual-review is_safety must be boolean.")
        assessment = row["intelligence_assessment"]
        if assessment not in {
            "useful",
            "partial",
            "incoherent",
            "censored",
            "not_scored",
        }:
            raise Lfm2MatrixCellError("Manual-review assessment is invalid.")
        if row["is_safety"] and assessment != "not_scored":
            raise Lfm2MatrixCellError(
                "Safety observations must remain outside intelligence scoring."
            )
        termination = row["termination_reason"]
        if termination not in {"eos", "cap", "loop", "error"}:
            raise Lfm2MatrixCellError("Manual-review termination is invalid.")
        censored = row["termination_censored"]
        if not isinstance(censored, bool) or (termination == "cap") != censored:
            raise Lfm2MatrixCellError(
                "Only cap termination is an endpoint-censored observation."
            )
        rows.append(
            {
                "prompt_id": prompt_id,
                "is_safety": row["is_safety"],
                "intelligence_assessment": assessment,
                "termination_reason": termination,
                "termination_censored": censored,
                "rationale": _text(
                    row["rationale"], f"prompt_reviews[{index}].rationale"
                ),
            }
        )
    if not rows:
        raise Lfm2MatrixCellError("Manual review needs at least one prompt row.")
    previous_sha256 = current["receipt_sha256"]
    archive = path.parent / "receipt-revisions" / f"{previous_sha256}.json"
    if not archive.exists():
        _atomic_write_json(archive, _plain_json(current, "receipt archive"))
    body = {
        key: _plain_json(value, f"receipt.{key}")
        for key, value in current.items()
        if key != "receipt_sha256"
    }
    body["manual_intelligence_review"] = {
        "status": status,
        "reviewer_kind": reviewer_kind,
        "reviewed_at_utc": _now(),
        "summary": _text(summary, "manual-review summary"),
        "prompt_reviews": rows,
        "whole_decoded_responses_authoritative": True,
        "exact_token_match_required": False,
        "safety_scored_separately": True,
        "cap_termination_is_censored_not_failed": True,
    }
    body["supersedes_receipt_sha256"] = previous_sha256
    body["automatic_intelligence_verdict_performed"] = False
    revised = {**body, "receipt_sha256": canonical_sha256(body)}
    _atomic_write_json(path, revised)
    authenticated = authenticate_receipt(revised)
    if ledger_path is not None:
        append_matrix_ledger(ledger_path, authenticated)
    return authenticated


def run_matrix_cell(
    request: MatrixCellRequest,
    executor: MatrixTechniqueExecutor,
    run_directory: str | os.PathLike[str],
    *,
    ledger_path: str | os.PathLike[str] | None = None,
) -> Mapping[str, Any]:
    """Execute, seal, and optionally ledger one component matrix cell."""

    if not isinstance(request, MatrixCellRequest) or not isinstance(
        executor, MatrixTechniqueExecutor
    ):
        raise Lfm2MatrixCellError("run_matrix_cell received invalid objects.")
    if executor.technique_id != request.technique_id:
        raise Lfm2MatrixCellError("Executor technique differs from the request.")
    root = Path(run_directory).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise Lfm2MatrixCellError("Run directory cannot be a symlink.")
    receipt_path = root / "cell-receipt.json"
    if receipt_path.exists():
        existing = authenticate_receipt(
            json.loads(receipt_path.read_text(encoding="utf-8"))
        )
        if existing["request_sha256"] != request.fingerprint:
            raise Lfm2MatrixCellError("Existing receipt belongs to another request.")
        return existing

    progress = MatrixProgressLog(root / "progress.jsonl", request.fingerprint)
    started = dt.datetime.now(dt.timezone.utc)
    attempt_index = sum(stage == "run_started" for stage in progress.stages)
    attempt_id = f"attempt-{attempt_index:04d}"
    attempt_event_offset = len(progress.events)
    progress.emit(
        "run_started",
        cell_id=request.cell_id,
        instance_id=request.instance_id,
        attempt_id=attempt_id,
    )
    progress.emit(
        "request_authenticated",
        request_sha256=request.fingerprint,
        evidence_sha256=request.evidence_binding["evidence_sha256"],
        evidence_class=request.evidence_class,
    )
    progress.emit(
        "executor_bound",
        executor_sha256=executor.fingerprint,
        implementation_sha256=executor.implementation_sha256,
    )
    try:
        raw = executor.execute(request, root, progress.emit)
        result = _validate_backend_result(raw)
        _validate_evidence_class_result(request, result)
        attempt_stages = progress.stages[attempt_event_offset:]
        missing_stages = [
            value for value in _REQUIRED_BACKEND_STAGES if value not in attempt_stages
        ]
        if missing_stages:
            raise Lfm2MatrixCellError(
                "Technique executor omitted required progress stages: "
                + ", ".join(missing_stages)
                + "."
            )
    except Exception as exc:
        progress.emit(
            "run_failed",
            attempt_id=attempt_id,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
        )
        failure_body = {
            "schema": FAILURE_SCHEMA,
            "status": "failed",
            "attempt_id": attempt_id,
            "request": request.as_dict(),
            "request_sha256": request.fingerprint,
            "executor": executor.as_dict(),
            "executor_sha256": executor.fingerprint,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "progress_event_count": len(progress.events),
            "progress_sha256": progress.fingerprint,
            "failed_at_utc": _now(),
            "resume_may_reuse_authenticated_checkpoints": True,
        }
        failure = {
            **failure_body,
            "failure_sha256": canonical_sha256(failure_body),
        }
        _atomic_write_json(root / "last-failure.json", failure)
        raise
    progress.emit(
        "result_authenticated",
        result_sha256=canonical_sha256(result),
        diagnostic_first_divergence=_diagnostic_first_divergence(result),
    )
    ended = dt.datetime.now(dt.timezone.utc)
    body = {
        "schema": RECEIPT_SCHEMA,
        "status": "complete",
        "attempt_id": attempt_id,
        "request": request.as_dict(),
        "request_sha256": request.fingerprint,
        "executor": executor.as_dict(),
        "executor_sha256": executor.fingerprint,
        "result": result,
        "result_sha256": canonical_sha256(result),
        "progress_event_count": len(progress.events),
        "progress_sha256_before_publication": progress.fingerprint,
        "started_at_utc": started.isoformat(timespec="milliseconds"),
        "completed_at_utc": ended.isoformat(timespec="milliseconds"),
        "wall_time_seconds": (ended - started).total_seconds(),
        "numeric_triage": {
            "first_divergent_metric": _diagnostic_first_divergence(result),
            "decision_authority": False,
            "purpose": "failure-localisation-only",
        },
        "manual_intelligence_review": {
            "status": "pending",
            "whole_decoded_responses_authoritative": True,
            "exact_token_match_required": False,
            "safety_scored_separately": True,
            "cap_termination_is_censored_not_failed": True,
        },
        "automatic_intelligence_verdict_performed": False,
        "compaction_claim_permitted": request.compaction_claim_permitted,
        "isolation_only_cannot_prove_compaction": (
            request.evidence_class == "isolation_only"
        ),
    }
    receipt = {**body, "receipt_sha256": canonical_sha256(body)}
    _atomic_write_json(receipt_path, receipt)
    progress.emit(
        "receipt_published",
        receipt_sha256=receipt["receipt_sha256"],
        receipt_path=str(receipt_path),
    )
    authenticated = authenticate_receipt(receipt)
    if ledger_path is not None:
        append_matrix_ledger(ledger_path, authenticated)
    return authenticated


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    previous: str | None = None
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Lfm2MatrixCellError("Matrix ledger JSONL is malformed.") from exc
        record_sha = row.pop("record_sha256", None)
        if (
            row.get("schema") != LEDGER_SCHEMA
            or row.get("sequence") != index
            or row.get("previous_record_sha256") != previous
            or canonical_sha256(row) != record_sha
        ):
            raise Lfm2MatrixCellError("Matrix ledger authentication failed.")
        row["record_sha256"] = record_sha
        records.append(row)
        previous = record_sha
    return records


def read_matrix_ledger(path: str | os.PathLike[str]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        MappingProxyType(value)
        for value in _read_ledger(Path(path).expanduser().absolute())
    )


def append_matrix_ledger(
    path: str | os.PathLike[str], receipt: Mapping[str, Any]
) -> Mapping[str, Any]:
    authenticated = authenticate_receipt(receipt)
    destination = Path(path).expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_file(destination):
        records = _read_ledger(destination)
        for value in records:
            if value["receipt_sha256"] == authenticated["receipt_sha256"]:
                return MappingProxyType(value)
        request = authenticated["request"]
        result = authenticated["result"]
        body = {
            "schema": LEDGER_SCHEMA,
            "sequence": len(records),
            "previous_record_sha256": (
                None if not records else records[-1]["record_sha256"]
            ),
            "recorded_at_utc": _now(),
            "instance_id": request["instance_id"],
            "cell_id": request["cell_id"],
            "component_ids": request["component_ids"],
            "technique_id": request["technique_id"],
            "level_id": request["level_id"],
            "site_id": request["site_id"],
            "evidence_class": request["evidence_class"],
            "execution_status": authenticated["status"],
            "manual_intelligence_status": authenticated[
                "manual_intelligence_review"
            ]["status"],
            "numeric_first_divergence": authenticated["numeric_triage"][
                "first_divergent_metric"
            ],
            "free_run_token_match_vs_control": result["behavior"]["free_run"][
                "token_match_vs_control"
            ],
            "free_run_mean_lcp": result["behavior"]["free_run"]["mean_lcp"],
            "terminal_heldout_rel_l2": result["composed"][
                "terminal_heldout_rel_l2"
            ],
            "receipt_sha256": authenticated["receipt_sha256"],
        }
        row = {**body, "record_sha256": canonical_sha256(body)}
        with destination.open("ab") as handle:
            handle.write(canonical_json_bytes(row))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    return MappingProxyType(row)


def wave_one_cell_ids() -> tuple[str, ...]:
    """The predeclared first-wave cells, without inventing new experiments."""

    return (
        "CTRL.copy_exact",
        "CAL.C01.T5.X4",
        "C05.T5.M16.L00",
        "C05.T5.M16.L05",
        "C05.T5.M16.L13",
        "C13.T5.M16.L00",
        "C13.T5.M16.L05",
        "C13.T5.M16.L13",
        "C11.T5.M16.L00",
        "C12.T5.M16.L00",
        "C03.T5.M16.L00",
        "C08.T5.M16.L02",
        "C09.T5.M16.L02",
        "C05.T1.M16.L00",
        "C05.T1.M16.L05",
        "C05.T1.M16.L13",
        "C13.T1.M16.L00",
        "C13.T1.M16.L05",
        "C06.T1.M16.L02",
        "C07.T1.M16.L02",
        "C08.T1.M16.L02",
        "C09.T1.M16.L02",
        "C04.T1.M16.L00",
        "C05.T8.M16.L00",
        "C13.T8.M16.L00",
        "C09.T8.M16.L02",
        "C06+C07.T7.M16.L02",
        "C06+C07.T7.M16.L08",
        "C05.T4.M16.L13",
        "C09.T4.M16.L02",
        "C10.T1.M16.L02",
        "C14.T1.NA.L00",
    )


__all__ = [
    "EXECUTOR_SCHEMA",
    "FAILURE_SCHEMA",
    "LEDGER_SCHEMA",
    "PROGRESS_SCHEMA",
    "RECEIPT_SCHEMA",
    "REQUEST_SCHEMA",
    "Lfm2MatrixCellError",
    "MatrixCellRequest",
    "MatrixProgressLog",
    "MatrixTechniqueExecutor",
    "append_matrix_ledger",
    "authenticate_receipt",
    "build_matrix_phase_key",
    "load_executor",
    "read_matrix_ledger",
    "record_manual_intelligence_review",
    "run_matrix_cell",
    "source_bound_executor",
    "wave_one_cell_ids",
]
