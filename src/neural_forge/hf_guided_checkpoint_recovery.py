"""Authenticated persistence for deterministic guided-checkpoint recovery.

The behavior trainer already emits immutable :class:`DurableBehaviorCheckpoint`
events.  This module is a deliberately separate callback sink which verifies
those events against historical checkpoint evidence, retains at most three
preregistered full v3 factor states, and atomically publishes a research-only
artifact tree after the trainer's terminal checkpoint rows agree with every
callback.

Historical trainer reports did not contain durable-checkpoint hashes or the
later per-tensor manifests.  Consequently ``expected_checkpoint_sha256`` is
optional on historical expectations.  The sink always authenticates and
records the newly generated durable checkpoint hash, but never describes it as
historically prebound when the expectation omitted it.
"""

from __future__ import annotations

import hashlib
import json
import operator
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch import Tensor

from neural_forge.hf_target_native_behavior_distillation import (
    BEHAVIOR_CHECKPOINT_SCHEMA,
    BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
    CausalBehaviorMetrics,
    DurableBehaviorCheckpoint,
    GuidedFactorSourceBinding,
)


GUIDED_CHECKPOINT_RECOVERY_SCHEMA = "neural-forge-guided-checkpoint-recovery-v1"
GUIDED_CHECKPOINT_CANDIDATE_SCHEMA = (
    "neural-forge-guided-checkpoint-recovery-candidate-v1"
)
GUIDED_CHECKPOINT_RESEARCH_MANIFEST_SCHEMA = (
    "neural-forge-guided-checkpoint-research-manifest-v1"
)
GUIDED_CHECKPOINT_PORTABLE_FACTOR_IDENTITY_SCHEMA = (
    "neural-forge-guided-checkpoint-portable-factor-identity-v1"
)

_FACTOR_PACK_FILENAME = "factor_pack.pt"
_CANDIDATE_RECORD_FILENAME = "candidate_record.json"
_RESEARCH_MANIFEST_FILENAME = "research_manifest.json"
_RETAINED_DIRECTORY = "retained"
_MAX_RETAINED_STEPS = 3
_SHA256_LENGTH = 64

_QKVO_GUD_FACTOR_NAMES = frozenset(
    {
        *(
            f"qkvo.{role}_{side}"
            for role in ("q", "k", "v", "o")
            for side in ("left", "right")
        ),
        *(
            f"gud.{role}_{side}"
            for role in ("gate", "up", "down")
            for side in ("left", "right")
        ),
    }
)
_LEXICAL_FACTOR_NAMES = frozenset(
    f"lexical.{role}_{side}"
    for role in ("embedding", "lm_head")
    for side in ("left", "right")
)
_GUIDED_QKVO_FACTOR_NAMES = frozenset(
    f"guided_qkvo.{role}_{side}"
    for role in ("q", "k", "v", "o")
    for side in ("left", "right")
)
_V3_FACTOR_NAMES = (
    _QKVO_GUD_FACTOR_NAMES | _LEXICAL_FACTOR_NAMES | _GUIDED_QKVO_FACTOR_NAMES
)

_METRIC_FIELDS = frozenset(CausalBehaviorMetrics.__dataclass_fields__)
_TERMINAL_CHECKPOINT_FIELDS = {
    "schema",
    "step",
    "metrics",
    "factor_state_sha256",
    "pareto_safe_vs_step0",
    "pareto_frontier",
    "strict_selection_key",
    "rollout_first_acquisition_key",
}
_V3_FACTOR_PAYLOAD_FIELDS = {
    "schema",
    "factor_count",
    "factors",
    "factor_state_sha256",
    "parent_state_included",
    "native_child_state_included",
    "optimizer_state_included",
    "lexical_rank",
    "guided_source_binding",
    "guided_source_binding_sha256",
    "inherited_product_regrouped",
    "guided_product_regrouped",
    "factor_tensors",
}
_RUNTIME_SOURCE_BINDING_FIELDS = frozenset({"inherited_installation_sha256"})


class GuidedCheckpointRecoveryError(RuntimeError):
    """Base class for recovery authentication or publication failures."""


class GuidedCheckpointRecoveryAuthenticationError(GuidedCheckpointRecoveryError):
    """A preregistration, callback, factor pack, or terminal row is invalid."""


class GuidedCheckpointRecoveryOrderError(GuidedCheckpointRecoveryError):
    """Callbacks are missing, duplicated, or outside the preregistered order."""


class GuidedCheckpointRecoveryPublicationError(GuidedCheckpointRecoveryError):
    """An owned staging artifact could not be atomically published."""


class GuidedCheckpointRecoveryStateError(GuidedCheckpointRecoveryError):
    """The sink lifecycle does not permit the requested operation."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the project's canonical JSON representation."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise GuidedCheckpointRecoveryAuthenticationError(
            "Recovery evidence is not canonical JSON."
        ) from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path | str) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: Any, name: str) -> str:
    result = str(value).strip().lower()
    if len(result) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise GuidedCheckpointRecoveryAuthenticationError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return result


def _step(value: Any, name: str = "step") -> int:
    if isinstance(value, bool):
        raise GuidedCheckpointRecoveryAuthenticationError(
            f"{name} must be a nonnegative integer."
        )
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GuidedCheckpointRecoveryAuthenticationError(
            f"{name} must be a nonnegative integer."
        ) from error
    if result < 0:
        raise GuidedCheckpointRecoveryAuthenticationError(
            f"{name} must be a nonnegative integer."
        )
    return result


def _tensor_sha256(value: Tensor) -> str:
    if not isinstance(value, Tensor):
        raise GuidedCheckpointRecoveryAuthenticationError(
            "A persisted factor value is not a tensor."
        )
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    flat = tensor.reshape(-1).view(torch.uint8)
    for start in range(0, int(flat.numel()), 1 << 20):
        digest.update(memoryview(flat[start : start + (1 << 20)].numpy()))
    return digest.hexdigest()


def _source_binding(value: Any, name: str) -> GuidedFactorSourceBinding:
    if isinstance(value, GuidedFactorSourceBinding):
        return value
    if not isinstance(value, Mapping):
        raise GuidedCheckpointRecoveryAuthenticationError(
            f"{name} must be a complete guided source binding."
        )
    try:
        return GuidedFactorSourceBinding(**dict(value))
    except (TypeError, ValueError, RuntimeError) as error:
        raise GuidedCheckpointRecoveryAuthenticationError(
            f"{name} failed trainer authentication."
        ) from error


def portable_guided_source_binding(
    binding: GuidedFactorSourceBinding,
) -> dict[str, Any]:
    """Return content provenance without the process-local installation receipt.

    ``inherited_installation_sha256`` authenticates a *live installation* and
    includes process-local tensor ownership state.  Every content-bearing input
    to that installation remains bound here; only the runtime receipt itself is
    intentionally kept outside portable checkpoint identity.
    """

    value = _source_binding(binding, "guided source binding")
    payload = value.as_dict()
    for name in _RUNTIME_SOURCE_BINDING_FIELDS:
        payload.pop(name)
    return payload


def portable_guided_source_binding_sha256(
    binding: GuidedFactorSourceBinding,
) -> str:
    return canonical_sha256(
        {
            "schema": GUIDED_CHECKPOINT_PORTABLE_FACTOR_IDENTITY_SCHEMA,
            "portable_guided_source_binding": portable_guided_source_binding(binding),
            "runtime_fields_excluded": sorted(_RUNTIME_SOURCE_BINDING_FIELDS),
        }
    )


def _legacy_v3_factor_state_sha256(
    payload: Mapping[str, Any],
    binding: GuidedFactorSourceBinding,
) -> str:
    """Reproduce the trainer's legacy v3 digest for exact historical evidence."""

    return canonical_sha256(
        {
            "schema": BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
            "lexical_rank": payload.get("lexical_rank"),
            "factors": payload.get("factors"),
            "guided_source_binding": binding.as_dict(),
            "guided_source_binding_sha256": binding.binding_sha256,
            "inherited_product_regrouped": False,
            "guided_product_regrouped": False,
        }
    )


def portable_v3_factor_state_sha256(
    payload: Mapping[str, Any],
    binding: GuidedFactorSourceBinding | None = None,
) -> str:
    """Content address for a v3 state that is stable across live processes."""

    source = _source_binding(
        payload.get("guided_source_binding") if binding is None else binding,
        "portable factor source binding",
    )
    return canonical_sha256(
        {
            "schema": GUIDED_CHECKPOINT_PORTABLE_FACTOR_IDENTITY_SCHEMA,
            "legacy_factor_pack_schema": BEHAVIOR_FACTOR_PACK_V3_SCHEMA,
            "lexical_rank": payload.get("lexical_rank"),
            "factors": payload.get("factors"),
            "portable_guided_source_binding_sha256": (
                portable_guided_source_binding_sha256(source)
            ),
            "inherited_product_regrouped": False,
            "guided_product_regrouped": False,
        }
    )


def _rebind_v3_factor_payload(
    payload: Mapping[str, Any],
    binding: GuidedFactorSourceBinding,
) -> dict[str, Any]:
    rebound = dict(payload)
    rebound["guided_source_binding"] = binding.as_dict()
    rebound["guided_source_binding_sha256"] = binding.binding_sha256
    rebound["factor_state_sha256"] = _legacy_v3_factor_state_sha256(rebound, binding)
    return rebound


def _write_all(handle: Any, payload: bytes) -> None:
    written = handle.write(payload)
    if written is not None and written != len(payload):
        raise GuidedCheckpointRecoveryPublicationError(
            "An atomic recovery write was incomplete."
        )
    handle.flush()
    os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=False, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            _write_all(handle, canonical_json_bytes(dict(payload)) + b"\n")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_save_factor_pack(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=False, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GuidedCheckpointRecoveryAuthenticationError(
            f"Recovery JSON {path.name!r} cannot be authenticated."
        ) from error
    if not isinstance(value, dict):
        raise GuidedCheckpointRecoveryAuthenticationError(
            f"Recovery JSON {path.name!r} must contain one object."
        )
    canonical_json_bytes(value)
    return value


@dataclass(frozen=True, slots=True)
class HistoricalCheckpointExpectation:
    """Historically available evidence for one ordered replay checkpoint."""

    step: int
    metrics: Mapping[str, Any]
    metrics_sha256: str
    factor_state_sha256: str
    expected_checkpoint_sha256: str | None = None
    metrics_report_sha256: str = field(init=False)
    expectation_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        step = _step(self.step, "historical checkpoint step")
        if not isinstance(self.metrics, Mapping):
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Historical checkpoint metrics must be a mapping."
            )
        metrics = dict(self.metrics)
        if set(metrics) != _METRIC_FIELDS:
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Historical checkpoint metrics are not the complete canonical "
                "CausalBehaviorMetrics payload."
            )
        try:
            reconstructed = CausalBehaviorMetrics(**metrics)
        except (TypeError, ValueError) as error:
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Historical checkpoint metrics cannot be reconstructed."
            ) from error
        if canonical_json_bytes(reconstructed.as_dict()) != canonical_json_bytes(
            metrics
        ):
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Historical checkpoint metrics changed during reconstruction."
            )
        metrics_digest = _sha256(self.metrics_sha256, "metrics_sha256")
        if canonical_sha256(metrics) != metrics_digest:
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Historical checkpoint metrics SHA does not authenticate its payload."
            )
        report_digest = _sha256(metrics.get("report_sha256"), "metrics.report_sha256")
        factor_digest = _sha256(self.factor_state_sha256, "factor_state_sha256")
        checkpoint_digest = (
            None
            if self.expected_checkpoint_sha256 is None
            else _sha256(
                self.expected_checkpoint_sha256,
                "expected_checkpoint_sha256",
            )
        )
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        object.__setattr__(self, "metrics_sha256", metrics_digest)
        object.__setattr__(self, "metrics_report_sha256", report_digest)
        object.__setattr__(self, "factor_state_sha256", factor_digest)
        object.__setattr__(self, "expected_checkpoint_sha256", checkpoint_digest)
        object.__setattr__(
            self,
            "expectation_sha256",
            canonical_sha256(self.as_dict()),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "metrics": dict(self.metrics),
            "metrics_sha256": self.metrics_sha256,
            "metrics_report_sha256": self.metrics_report_sha256,
            "factor_state_sha256": self.factor_state_sha256,
            "expected_checkpoint_sha256": self.expected_checkpoint_sha256,
            "durable_checkpoint_sha256_historically_available": (
                self.expected_checkpoint_sha256 is not None
            ),
        }


@dataclass(frozen=True, slots=True)
class GuidedCheckpointRecoveryPublication:
    """Paths and hashes returned after one successful atomic publication."""

    output_root: Path
    manifest_path: Path
    manifest_sha256: str
    manifest_file_sha256: str
    retained_steps: tuple[int, ...]
    retained_factor_state_sha256s: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_root": str(self.output_root),
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "manifest_file_sha256": self.manifest_file_sha256,
            "retained_steps": list(self.retained_steps),
            "retained_factor_state_sha256s": list(self.retained_factor_state_sha256s),
        }


@dataclass(frozen=True, slots=True)
class _ObservedCheckpoint:
    step: int
    metrics: Mapping[str, Any]
    metrics_sha256: str
    metrics_report_sha256: str
    factor_state_sha256: str
    runtime_factor_state_sha256: str
    legacy_rebound_factor_state_sha256: str
    portable_factor_state_sha256: str
    runtime_guided_source_binding_sha256: str
    runtime_inherited_installation_sha256: str
    durable_checkpoint_sha256: str
    historical_checkpoint_sha256: str | None
    expectation_sha256: str
    pareto_safe_vs_step0: bool
    strict_selection_key: tuple[float, ...]
    rollout_first_acquisition_key: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "metrics": dict(self.metrics),
            "metrics_sha256": self.metrics_sha256,
            "metrics_report_sha256": self.metrics_report_sha256,
            "factor_state_sha256": self.factor_state_sha256,
            "runtime_factor_state_sha256": self.runtime_factor_state_sha256,
            "legacy_rebound_factor_state_sha256": (
                self.legacy_rebound_factor_state_sha256
            ),
            "portable_factor_state_sha256": self.portable_factor_state_sha256,
            "runtime_guided_source_binding_sha256": (
                self.runtime_guided_source_binding_sha256
            ),
            "runtime_inherited_installation_sha256": (
                self.runtime_inherited_installation_sha256
            ),
            "durable_checkpoint_sha256": self.durable_checkpoint_sha256,
            "historical_checkpoint_sha256": self.historical_checkpoint_sha256,
            "durable_checkpoint_sha256_historically_prebound": (
                self.historical_checkpoint_sha256 is not None
            ),
            "expectation_sha256": self.expectation_sha256,
            "pareto_safe_vs_step0": self.pareto_safe_vs_step0,
            "strict_selection_key": list(self.strict_selection_key),
            "rollout_first_acquisition_key": list(self.rollout_first_acquisition_key),
        }


class GuidedCheckpointRecoverySink:
    """Callable recovery sink with fail-closed, directory-atomic publication."""

    def __init__(
        self,
        output_root: Path | str,
        *,
        expectations: Sequence[HistoricalCheckpointExpectation],
        retained_steps: Sequence[int],
        recovery_preregistration_sha256: str,
        recovery_report_sha256: str,
        source_run_report_sha256: str,
        historical_guided_source_binding: GuidedFactorSourceBinding,
    ) -> None:
        values = tuple(expectations)
        if not values or any(
            not isinstance(value, HistoricalCheckpointExpectation) for value in values
        ):
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Recovery requires ordered HistoricalCheckpointExpectation values."
            )
        steps = tuple(value.step for value in values)
        if len(set(steps)) != len(steps) or steps != tuple(sorted(steps)):
            raise GuidedCheckpointRecoveryOrderError(
                "Historical checkpoint expectations must be strictly increasing."
            )
        retained = tuple(_step(value, "retained step") for value in retained_steps)
        if (
            not retained
            or len(retained) > _MAX_RETAINED_STEPS
            or len(set(retained)) != len(retained)
            or retained != tuple(step for step in steps if step in set(retained))
        ):
            raise GuidedCheckpointRecoveryOrderError(
                "Retained steps must be one to three unique expectation steps in "
                "preregistered order."
            )

        output = Path(output_root).expanduser().resolve()
        if output.exists() or not output.name or output.name in {".", ".."}:
            raise GuidedCheckpointRecoveryPublicationError(
                "Recovery output_root must be one absent named directory."
            )
        parent = output.parent
        if not parent.is_dir() or parent.is_symlink():
            raise GuidedCheckpointRecoveryPublicationError(
                "Recovery output parent must be an existing real directory."
            )

        self.output_root = output
        self.expectations = values
        self.retained_steps = retained
        self.recovery_preregistration_sha256 = _sha256(
            recovery_preregistration_sha256,
            "recovery_preregistration_sha256",
        )
        self.recovery_report_sha256 = _sha256(
            recovery_report_sha256, "recovery_report_sha256"
        )
        self.source_run_report_sha256 = _sha256(
            source_run_report_sha256, "source_run_report_sha256"
        )
        self.historical_guided_source_binding = _source_binding(
            historical_guided_source_binding,
            "historical guided source binding",
        )
        self.portable_guided_source_binding_sha256 = (
            portable_guided_source_binding_sha256(self.historical_guided_source_binding)
        )
        self._expectation_by_step = {value.step: value for value in values}
        self._observed: list[_ObservedCheckpoint] = []
        self._observed_steps: set[int] = set()
        self._retained_step_to_factor: dict[int, str] = {}
        self._failure_diagnostic: dict[str, Any] | None = None
        self._state: Literal["collecting", "finalized", "failed", "cleaned"] = (
            "collecting"
        )
        self._staging_root = Path(
            tempfile.mkdtemp(
                prefix=f".{output.name}.guided-recovery-",
                suffix=".staging",
                dir=parent,
            )
        )
        self._owned_staging_root = self._staging_root.resolve()
        self._retained_root = self._staging_root / _RETAINED_DIRECTORY
        self._retained_root.mkdir()

    @property
    def state(self) -> str:
        return self._state

    @property
    def staging_root(self) -> Path | None:
        """Expose the current owned root for monitoring, never for cleanup input."""

        return self._staging_root

    @property
    def failure_diagnostic(self) -> Mapping[str, Any] | None:
        """Return immutable observed-vs-expected evidence captured before cleanup."""

        if self._failure_diagnostic is None:
            return None
        return MappingProxyType(dict(self._failure_diagnostic))

    def _assert_collecting(self) -> None:
        if self._state != "collecting":
            raise GuidedCheckpointRecoveryStateError(
                f"Recovery sink is {self._state}, not collecting."
            )

    def _safe_cleanup_staging(self) -> None:
        root = self._staging_root
        if root is None or not root.exists():
            self._staging_root = None
            return
        resolved = root.resolve()
        expected_parent = self.output_root.parent.resolve()
        expected_prefix = f".{self.output_root.name}.guided-recovery-"
        if (
            resolved != self._owned_staging_root
            or resolved.parent != expected_parent
            or not resolved.name.startswith(expected_prefix)
            or not resolved.name.endswith(".staging")
        ):
            raise GuidedCheckpointRecoveryStateError(
                "Refusing to clean a path outside the owned recovery staging root."
            )
        shutil.rmtree(resolved)
        self._staging_root = None

    def cleanup(self) -> None:
        """Remove only the constructor-created private staging root.

        The method accepts no path and never removes an already published output.
        """

        if self._state == "finalized":
            return
        self._safe_cleanup_staging()
        if self._state == "collecting":
            self._state = "cleaned"

    def _fail(self) -> None:
        self._state = "failed"
        self._safe_cleanup_staging()

    @staticmethod
    def _validate_v3_factor_payload(
        payload: Mapping[str, Any],
        *,
        expected_factor_state_sha256: str,
    ) -> tuple[dict[str, Any], str]:
        if set(payload) != _V3_FACTOR_PAYLOAD_FIELDS:
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Durable factor payload is not the exact v3 schema."
            )
        raw_binding = payload.get("guided_source_binding")
        if not isinstance(raw_binding, Mapping):
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Durable v3 factor payload has no guided source binding."
            )
        try:
            reconstructed_binding = GuidedFactorSourceBinding(**dict(raw_binding))
        except (TypeError, ValueError, RuntimeError) as error:
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Durable v3 guided source binding failed trainer authentication."
            ) from error
        if (
            payload.get("schema") != BEHAVIOR_FACTOR_PACK_V3_SCHEMA
            or payload.get("factor_count") != len(_V3_FACTOR_NAMES)
            or payload.get("factor_state_sha256") != expected_factor_state_sha256
            or any(
                payload.get(name) is not False
                for name in (
                    "parent_state_included",
                    "native_child_state_included",
                    "optimizer_state_included",
                    "inherited_product_regrouped",
                    "guided_product_regrouped",
                )
            )
            or isinstance(payload.get("lexical_rank"), bool)
            or not isinstance(payload.get("lexical_rank"), int)
            or payload.get("lexical_rank") <= 0
            or payload.get("guided_source_binding_sha256")
            != reconstructed_binding.binding_sha256
            or canonical_json_bytes(raw_binding)
            != canonical_json_bytes(reconstructed_binding.as_dict())
        ):
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Durable v3 factor payload metadata failed authentication."
            )
        rows = payload.get("factors")
        tensors = payload.get("factor_tensors")
        if (
            not isinstance(rows, list)
            or not isinstance(tensors, Mapping)
            or len(rows) != len(_V3_FACTOR_NAMES)
            or set(tensors) != _V3_FACTOR_NAMES
        ):
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Durable v3 factor payload is not the complete 26-factor state."
            )
        by_name: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {
                "name",
                "shape",
                "dtype",
                "tensor_sha256",
            }:
                raise GuidedCheckpointRecoveryAuthenticationError(
                    "Durable v3 factor manifest row is malformed."
                )
            name = row.get("name")
            if not isinstance(name, str) or name in by_name:
                raise GuidedCheckpointRecoveryAuthenticationError(
                    "Durable v3 factor names must be unique strings."
                )
            by_name[name] = row
        if set(by_name) != _V3_FACTOR_NAMES:
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Durable v3 factor manifest does not contain exactly 26 factors."
            )
        for name in sorted(_V3_FACTOR_NAMES):
            tensor = tensors[name]
            row = by_name[name]
            if (
                not isinstance(tensor, Tensor)
                or list(tensor.shape) != row.get("shape")
                or str(tensor.dtype) != row.get("dtype")
                or _tensor_sha256(tensor)
                != _sha256(row.get("tensor_sha256"), f"factor {name} SHA")
                or not bool(torch.isfinite(tensor.detach()).all())
            ):
                raise GuidedCheckpointRecoveryAuthenticationError(
                    f"Durable v3 factor {name!r} failed tensor authentication."
                )
        manifest = {
            name: value for name, value in payload.items() if name != "factor_tensors"
        }
        return manifest, canonical_sha256(manifest)

    def _authenticate_event(
        self,
        event: DurableBehaviorCheckpoint,
        expectation: HistoricalCheckpointExpectation,
    ) -> tuple[_ObservedCheckpoint, dict[str, Any], str]:
        if not isinstance(event, DurableBehaviorCheckpoint):
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Recovery callback requires DurableBehaviorCheckpoint events."
            )
        manifest = event.as_manifest()
        durable_checkpoint_sha = _sha256(
            event.checkpoint_sha256, "event.checkpoint_sha256"
        )
        if canonical_sha256(manifest) != durable_checkpoint_sha:
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Durable callback checkpoint SHA failed self-authentication."
            )
        if (
            expectation.expected_checkpoint_sha256 is not None
            and durable_checkpoint_sha != expectation.expected_checkpoint_sha256
        ):
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Durable callback checkpoint SHA differs from its optional "
                "historical pin."
            )
        metrics = event.metrics.as_dict()
        metrics_sha = canonical_sha256(metrics)
        report_sha = _sha256(metrics.get("report_sha256"), "metrics.report_sha256")
        if (
            canonical_json_bytes(metrics)
            != canonical_json_bytes(dict(expectation.metrics))
            or metrics_sha != expectation.metrics_sha256
            or report_sha != expectation.metrics_report_sha256
        ):
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Durable callback metrics differ from historical replay evidence."
            )
        factor_sha = _sha256(
            event.factor_state.factor_state_sha256,
            "event.factor_state_sha256",
        )
        try:
            factor_payload = event.factor_state.as_factor_payload()
        except Exception as error:
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Durable callback factor bytes failed materialization authentication."
            ) from error
        factor_manifest, factor_manifest_sha = self._validate_v3_factor_payload(
            factor_payload,
            expected_factor_state_sha256=factor_sha,
        )
        runtime_binding = _source_binding(
            factor_payload.get("guided_source_binding"),
            "runtime guided source binding",
        )
        portable_binding_matches = canonical_json_bytes(
            portable_guided_source_binding(runtime_binding)
        ) == canonical_json_bytes(
            portable_guided_source_binding(self.historical_guided_source_binding)
        )
        legacy_rebound_payload = _rebind_v3_factor_payload(
            factor_payload,
            self.historical_guided_source_binding,
        )
        legacy_rebound_sha = _sha256(
            legacy_rebound_payload["factor_state_sha256"],
            "legacy rebound factor state",
        )
        portable_factor_sha = portable_v3_factor_state_sha256(
            factor_payload,
            runtime_binding,
        )
        if not portable_binding_matches or (
            legacy_rebound_sha != expectation.factor_state_sha256
        ):
            self._failure_diagnostic = {
                "schema": "neural-forge-guided-checkpoint-recovery-failure-v1",
                "step": expectation.step,
                "failure": "historical-factor-state-mismatch",
                "expected_factor_state_sha256": expectation.factor_state_sha256,
                "observed_runtime_factor_state_sha256": factor_sha,
                "observed_legacy_rebound_factor_state_sha256": legacy_rebound_sha,
                "observed_portable_factor_state_sha256": portable_factor_sha,
                "portable_source_binding_matches": portable_binding_matches,
                "historical_guided_source_binding_sha256": (
                    self.historical_guided_source_binding.binding_sha256
                ),
                "runtime_guided_source_binding_sha256": (
                    runtime_binding.binding_sha256
                ),
                "historical_inherited_installation_sha256": (
                    self.historical_guided_source_binding.inherited_installation_sha256
                ),
                "runtime_inherited_installation_sha256": (
                    runtime_binding.inherited_installation_sha256
                ),
                "portable_guided_source_binding_sha256": (
                    self.portable_guided_source_binding_sha256
                ),
                "runtime_only_binding_fields": sorted(_RUNTIME_SOURCE_BINDING_FIELDS),
            }
            self._failure_diagnostic["diagnostic_sha256"] = canonical_sha256(
                self._failure_diagnostic
            )
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Durable callback tensors do not reproduce the historical legacy "
                "factor-state SHA under its authenticated source binding."
            )
        # Store the recovered bytes in their authenticated historical hash
        # domain; retain the current installation receipt only as runtime
        # evidence on the observed callback.
        factor_payload = legacy_rebound_payload
        factor_manifest, factor_manifest_sha = self._validate_v3_factor_payload(
            factor_payload,
            expected_factor_state_sha256=expectation.factor_state_sha256,
        )
        if not isinstance(event.pareto_safe_vs_step0, bool):
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Durable callback Pareto safety flag is not boolean."
            )
        observed = _ObservedCheckpoint(
            step=event.step,
            metrics=MappingProxyType(dict(metrics)),
            metrics_sha256=metrics_sha,
            metrics_report_sha256=report_sha,
            factor_state_sha256=expectation.factor_state_sha256,
            runtime_factor_state_sha256=factor_sha,
            legacy_rebound_factor_state_sha256=legacy_rebound_sha,
            portable_factor_state_sha256=portable_factor_sha,
            runtime_guided_source_binding_sha256=runtime_binding.binding_sha256,
            runtime_inherited_installation_sha256=(
                runtime_binding.inherited_installation_sha256
            ),
            durable_checkpoint_sha256=durable_checkpoint_sha,
            historical_checkpoint_sha256=(expectation.expected_checkpoint_sha256),
            expectation_sha256=expectation.expectation_sha256,
            pareto_safe_vs_step0=event.pareto_safe_vs_step0,
            strict_selection_key=tuple(event.metrics.selection_key),
            rollout_first_acquisition_key=tuple(
                event.metrics.rollout_first_acquisition_key
            ),
        )
        return observed, factor_payload, factor_manifest_sha

    def _candidate_event_row(self, observed: _ObservedCheckpoint) -> dict[str, Any]:
        return observed.as_dict()

    def _candidate_record(
        self,
        *,
        factor_state_sha256: str,
        portable_factor_state_sha256: str,
        factor_manifest_sha256: str,
        factor_pack_file_sha256: str,
        factor_pack_size_bytes: int,
        events: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": GUIDED_CHECKPOINT_CANDIDATE_SCHEMA,
            "recovery_preregistration_sha256": (self.recovery_preregistration_sha256),
            "recovery_report_sha256": self.recovery_report_sha256,
            "source_run_report_sha256": self.source_run_report_sha256,
            "factor_state_sha256": factor_state_sha256,
            "portable_factor_state_sha256": portable_factor_state_sha256,
            "portable_guided_source_binding_sha256": (
                self.portable_guided_source_binding_sha256
            ),
            "runtime_identity_excluded_from_portable_content": sorted(
                _RUNTIME_SOURCE_BINDING_FIELDS
            ),
            "factor_pack_relative_path": _FACTOR_PACK_FILENAME,
            "factor_pack_manifest_sha256": factor_manifest_sha256,
            "factor_pack_file_sha256": factor_pack_file_sha256,
            "factor_pack_size_bytes": factor_pack_size_bytes,
            "events": [dict(value) for value in events],
            "event_count": len(events),
            "retained_steps": [int(value["step"]) for value in events],
            "content_address": factor_state_sha256,
            "historical_durable_checkpoint_sha_required": False,
            "replay_durable_checkpoint_sha_self_authenticated": True,
            "contains_parent_state": False,
            "contains_native_child_state": False,
            "contains_optimizer_state": False,
            "contains_model_checkpoint": False,
        }
        payload["candidate_record_sha256"] = canonical_sha256(payload)
        return payload

    def _persist_retained_event(
        self,
        observed: _ObservedCheckpoint,
        factor_payload: Mapping[str, Any],
        factor_manifest_sha256: str,
    ) -> None:
        factor_sha = observed.factor_state_sha256
        destination = self._retained_root / factor_sha
        event_row = self._candidate_event_row(observed)
        if destination.exists():
            record_path = destination / _CANDIDATE_RECORD_FILENAME
            pack_path = destination / _FACTOR_PACK_FILENAME
            record = _json_file(record_path)
            declared_record_sha = _sha256(
                record.get("candidate_record_sha256"),
                "candidate_record_sha256",
            )
            if (
                canonical_sha256(
                    {
                        name: value
                        for name, value in record.items()
                        if name != "candidate_record_sha256"
                    }
                )
                != declared_record_sha
            ):
                raise GuidedCheckpointRecoveryAuthenticationError(
                    "Existing deduplicated candidate record failed authentication."
                )
            if (
                record.get("factor_state_sha256") != factor_sha
                or record.get("portable_factor_state_sha256")
                != observed.portable_factor_state_sha256
                or record.get("factor_pack_manifest_sha256") != factor_manifest_sha256
                or record.get("factor_pack_file_sha256") != file_sha256(pack_path)
                or record.get("factor_pack_size_bytes") != pack_path.stat().st_size
            ):
                raise GuidedCheckpointRecoveryAuthenticationError(
                    "Deduplicated factor artifact differs from the current event."
                )
            events = record.get("events")
            if not isinstance(events, list):
                raise GuidedCheckpointRecoveryAuthenticationError(
                    "Deduplicated candidate events are malformed."
                )
            if any(
                isinstance(value, Mapping) and value.get("step") == observed.step
                for value in events
            ):
                raise GuidedCheckpointRecoveryOrderError(
                    "A retained checkpoint step was duplicated."
                )
            updated = self._candidate_record(
                factor_state_sha256=factor_sha,
                portable_factor_state_sha256=(observed.portable_factor_state_sha256),
                factor_manifest_sha256=factor_manifest_sha256,
                factor_pack_file_sha256=str(record["factor_pack_file_sha256"]),
                factor_pack_size_bytes=int(record["factor_pack_size_bytes"]),
                events=(*events, event_row),
            )
            _atomic_write_json(record_path, updated)
            self._retained_step_to_factor[observed.step] = factor_sha
            return

        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".candidate-{factor_sha[:12]}-",
                suffix=".staging",
                dir=self._retained_root,
            )
        )
        try:
            pack_path = temporary / _FACTOR_PACK_FILENAME
            _atomic_save_factor_pack(pack_path, factor_payload)
            try:
                loaded = torch.load(pack_path, map_location="cpu", weights_only=True)
            except Exception as error:
                raise GuidedCheckpointRecoveryAuthenticationError(
                    "Persisted v3 factor pack cannot be safely reloaded."
                ) from error
            if not isinstance(loaded, Mapping):
                raise GuidedCheckpointRecoveryAuthenticationError(
                    "Persisted v3 factor pack did not reload as a mapping."
                )
            _, loaded_manifest_sha = self._validate_v3_factor_payload(
                loaded,
                expected_factor_state_sha256=factor_sha,
            )
            if loaded_manifest_sha != factor_manifest_sha256:
                raise GuidedCheckpointRecoveryAuthenticationError(
                    "Persisted v3 factor pack manifest changed after reload."
                )
            pack_file_sha = file_sha256(pack_path)
            record = self._candidate_record(
                factor_state_sha256=factor_sha,
                portable_factor_state_sha256=(observed.portable_factor_state_sha256),
                factor_manifest_sha256=factor_manifest_sha256,
                factor_pack_file_sha256=pack_file_sha,
                factor_pack_size_bytes=pack_path.stat().st_size,
                events=(event_row,),
            )
            _atomic_write_json(temporary / _CANDIDATE_RECORD_FILENAME, record)
            _fsync_directory(temporary)
            os.rename(temporary, destination)
            _fsync_directory(self._retained_root)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        self._retained_step_to_factor[observed.step] = factor_sha

    def __call__(self, event: DurableBehaviorCheckpoint) -> None:
        self._assert_collecting()
        try:
            if len(self._observed) >= len(self.expectations):
                raise GuidedCheckpointRecoveryOrderError(
                    "Recovery received a callback after the preregistered sequence."
                )
            expectation = self.expectations[len(self._observed)]
            event_step = _step(getattr(event, "step", None), "callback step")
            if event_step in self._observed_steps:
                raise GuidedCheckpointRecoveryOrderError(
                    f"Recovery callback step {event_step} was duplicated."
                )
            if event_step != expectation.step:
                raise GuidedCheckpointRecoveryOrderError(
                    f"Recovery expected step {expectation.step}, got {event_step}."
                )
            observed, factor_payload, factor_manifest_sha = self._authenticate_event(
                event, expectation
            )
            if observed.step in self.retained_steps:
                self._persist_retained_event(
                    observed,
                    factor_payload,
                    factor_manifest_sha,
                )
            self._observed.append(observed)
            self._observed_steps.add(observed.step)
        except BaseException:
            self._fail()
            raise

    def _authenticate_terminal_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], str]:
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Terminal checkpoint rows must be an ordered sequence."
            )
        values = tuple(rows)
        if len(values) != len(self._observed):
            raise GuidedCheckpointRecoveryOrderError(
                "Terminal checkpoint rows do not cover every callback."
            )
        canonical_rows: list[dict[str, Any]] = []
        for index, (row, observed) in enumerate(
            zip(values, self._observed, strict=True)
        ):
            if not isinstance(row, Mapping) or set(row) != _TERMINAL_CHECKPOINT_FIELDS:
                raise GuidedCheckpointRecoveryAuthenticationError(
                    f"Terminal checkpoint row {index} has an unsupported schema."
                )
            value = dict(row)
            runtime_factor_sha = value.get("factor_state_sha256")
            if (
                value.get("schema") != BEHAVIOR_CHECKPOINT_SCHEMA
                or value.get("step") != observed.step
                or canonical_json_bytes(value.get("metrics"))
                != canonical_json_bytes(dict(observed.metrics))
                or canonical_sha256(value.get("metrics")) != observed.metrics_sha256
                or runtime_factor_sha
                not in {
                    observed.factor_state_sha256,
                    observed.runtime_factor_state_sha256,
                }
                or value.get("pareto_safe_vs_step0")
                is not observed.pareto_safe_vs_step0
                or not isinstance(value.get("pareto_frontier"), bool)
                or value.get("strict_selection_key")
                != list(observed.strict_selection_key)
                or value.get("rollout_first_acquisition_key")
                != list(observed.rollout_first_acquisition_key)
            ):
                raise GuidedCheckpointRecoveryAuthenticationError(
                    f"Terminal checkpoint row {index} differs from its callback."
                )
            value["factor_state_sha256"] = observed.factor_state_sha256
            canonical_json_bytes(value)
            canonical_rows.append(value)
        return canonical_rows, canonical_sha256(canonical_rows)

    def normalize_runtime_trainer_evidence(
        self,
        evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Translate only runtime-bound v3 identities to the historical domain.

        All checkpoint tensor bytes were already authenticated by callbacks.
        This method therefore changes no metric, tensor manifest row, schedule,
        or training audit; it only substitutes the authenticated historical
        binding and the legacy digests reconstructed from that binding.
        """

        if not isinstance(evidence, Mapping):
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Runtime trainer evidence must be a mapping."
            )
        expected_steps = tuple(value.step for value in self.expectations)
        if tuple(value.step for value in self._observed) != expected_steps:
            raise GuidedCheckpointRecoveryOrderError(
                "Runtime trainer evidence cannot normalize before all callbacks."
            )
        try:
            normalized = json.loads(canonical_json_bytes(dict(evidence)))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Runtime trainer evidence could not be cloned canonically."
            ) from error
        runtime_binding = _source_binding(
            normalized.get("guided_source_binding"),
            "runtime trainer guided source binding",
        )
        observed_binding_shas = {
            value.runtime_guided_source_binding_sha256 for value in self._observed
        }
        if (
            observed_binding_shas != {runtime_binding.binding_sha256}
            or canonical_json_bytes(portable_guided_source_binding(runtime_binding))
            != canonical_json_bytes(
                portable_guided_source_binding(self.historical_guided_source_binding)
            )
            or normalized.get("guided_source_binding_sha256")
            != runtime_binding.binding_sha256
        ):
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Runtime trainer source binding differs from callback provenance."
            )
        terminal_rows, _terminal_sha = self._authenticate_terminal_rows(
            normalized.get("checkpoints")
        )
        normalized["checkpoints"] = terminal_rows

        step0 = self._observed[0]
        if normalized.get("combined_step0_factor_state_sha256") not in {
            step0.factor_state_sha256,
            step0.runtime_factor_state_sha256,
        }:
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Runtime trainer step-zero identity differs from its callback."
            )
        normalized["combined_step0_factor_state_sha256"] = step0.factor_state_sha256

        selected_step = _step(normalized.get("selected_step"), "selected step")
        selected = next(
            (value for value in self._observed if value.step == selected_step),
            None,
        )
        factor_manifest = normalized.get("factor_pack_manifest")
        if selected is None or not isinstance(factor_manifest, Mapping):
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Runtime trainer selected factor manifest is absent."
            )
        manifest_binding = _source_binding(
            factor_manifest.get("guided_source_binding"),
            "runtime selected factor binding",
        )
        if (
            manifest_binding != runtime_binding
            or factor_manifest.get("guided_source_binding_sha256")
            != runtime_binding.binding_sha256
            or factor_manifest.get("factor_state_sha256")
            not in {
                selected.factor_state_sha256,
                selected.runtime_factor_state_sha256,
            }
        ):
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Runtime selected factor manifest differs from callback provenance."
            )
        rebound_manifest = _rebind_v3_factor_payload(
            factor_manifest,
            self.historical_guided_source_binding,
        )
        if rebound_manifest["factor_state_sha256"] != selected.factor_state_sha256:
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Runtime selected tensor manifest does not reproduce its historical "
                "legacy identity."
            )
        normalized["factor_pack_manifest"] = rebound_manifest
        normalized["guided_source_binding"] = (
            self.historical_guided_source_binding.as_dict()
        )
        normalized["guided_source_binding_sha256"] = (
            self.historical_guided_source_binding.binding_sha256
        )
        return normalized

    def _retained_artifact_rows(self) -> list[dict[str, Any]]:
        expected_factors = {
            self._retained_step_to_factor[step] for step in self.retained_steps
        }
        actual_directories = {
            path.name
            for path in self._retained_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        }
        if actual_directories != expected_factors:
            raise GuidedCheckpointRecoveryAuthenticationError(
                "Retained content-addressed directories are missing or unexpected."
            )
        rows: list[dict[str, Any]] = []
        for factor_sha in sorted(expected_factors):
            directory = self._retained_root / factor_sha
            children = {path.name for path in directory.iterdir()}
            if children != {_FACTOR_PACK_FILENAME, _CANDIDATE_RECORD_FILENAME}:
                raise GuidedCheckpointRecoveryAuthenticationError(
                    "A retained candidate directory has missing or extra files."
                )
            pack_path = directory / _FACTOR_PACK_FILENAME
            record_path = directory / _CANDIDATE_RECORD_FILENAME
            record = _json_file(record_path)
            record_sha = _sha256(
                record.get("candidate_record_sha256"),
                "candidate_record_sha256",
            )
            if (
                canonical_sha256(
                    {
                        name: value
                        for name, value in record.items()
                        if name != "candidate_record_sha256"
                    }
                )
                != record_sha
            ):
                raise GuidedCheckpointRecoveryAuthenticationError(
                    "Retained candidate record self-hash changed."
                )
            steps = tuple(
                step
                for step in self.retained_steps
                if self._retained_step_to_factor[step] == factor_sha
            )
            if (
                record.get("factor_state_sha256") != factor_sha
                or not isinstance(record.get("portable_factor_state_sha256"), str)
                or record.get("retained_steps") != list(steps)
                or record.get("event_count") != len(steps)
                or record.get("factor_pack_file_sha256") != file_sha256(pack_path)
                or record.get("factor_pack_size_bytes") != pack_path.stat().st_size
            ):
                raise GuidedCheckpointRecoveryAuthenticationError(
                    "Retained candidate record no longer matches its artifact."
                )
            rows.append(
                {
                    "factor_state_sha256": factor_sha,
                    "portable_factor_state_sha256": record[
                        "portable_factor_state_sha256"
                    ],
                    "retained_steps": list(steps),
                    "directory": f"{_RETAINED_DIRECTORY}/{factor_sha}",
                    "factor_pack": (
                        f"{_RETAINED_DIRECTORY}/{factor_sha}/{_FACTOR_PACK_FILENAME}"
                    ),
                    "factor_pack_file_sha256": record["factor_pack_file_sha256"],
                    "factor_pack_size_bytes": record["factor_pack_size_bytes"],
                    "factor_pack_manifest_sha256": record[
                        "factor_pack_manifest_sha256"
                    ],
                    "candidate_record": (
                        f"{_RETAINED_DIRECTORY}/{factor_sha}/"
                        f"{_CANDIDATE_RECORD_FILENAME}"
                    ),
                    "candidate_record_sha256": record_sha,
                    "candidate_record_file_sha256": file_sha256(record_path),
                }
            )
        return rows

    def finalize(
        self,
        *,
        trainer_evidence_sha256: str,
        runtime_trainer_evidence_sha256: str | None = None,
        terminal_checkpoint_rows: Sequence[Mapping[str, Any]],
    ) -> GuidedCheckpointRecoveryPublication:
        self._assert_collecting()
        try:
            expected_steps = tuple(value.step for value in self.expectations)
            observed_steps = tuple(value.step for value in self._observed)
            if observed_steps != expected_steps:
                missing = [
                    step for step in expected_steps if step not in self._observed_steps
                ]
                raise GuidedCheckpointRecoveryOrderError(
                    f"Recovery cannot finalize with missing callbacks {missing}."
                )
            evidence_sha = _sha256(trainer_evidence_sha256, "trainer_evidence_sha256")
            runtime_evidence_sha = (
                evidence_sha
                if runtime_trainer_evidence_sha256 is None
                else _sha256(
                    runtime_trainer_evidence_sha256,
                    "runtime_trainer_evidence_sha256",
                )
            )
            terminal_rows, terminal_rows_sha = self._authenticate_terminal_rows(
                terminal_checkpoint_rows
            )
            if set(self._retained_step_to_factor) != set(self.retained_steps):
                raise GuidedCheckpointRecoveryOrderError(
                    "Not every retained step has an authenticated factor artifact."
                )
            retained_artifacts = self._retained_artifact_rows()
            payload: dict[str, Any] = {
                "schema": GUIDED_CHECKPOINT_RESEARCH_MANIFEST_SCHEMA,
                "status": "complete",
                "recovery_schema": GUIDED_CHECKPOINT_RECOVERY_SCHEMA,
                "recovery_preregistration_sha256": (
                    self.recovery_preregistration_sha256
                ),
                "recovery_report_sha256": self.recovery_report_sha256,
                "source_run_report_sha256": self.source_run_report_sha256,
                "trainer_evidence_sha256": evidence_sha,
                "runtime_trainer_evidence_sha256": runtime_evidence_sha,
                "trainer_evidence_runtime_identity_normalized": (
                    runtime_evidence_sha != evidence_sha
                ),
                "terminal_checkpoint_rows": terminal_rows,
                "terminal_checkpoint_rows_sha256": terminal_rows_sha,
                "expected_steps": list(expected_steps),
                "observed_events": [value.as_dict() for value in self._observed],
                "observed_event_count": len(self._observed),
                "retained_steps": list(self.retained_steps),
                "retained_step_to_factor_state_sha256": {
                    str(step): self._retained_step_to_factor[step]
                    for step in self.retained_steps
                },
                "retained_unique_factor_state_count": len(retained_artifacts),
                "maximum_retained_steps": _MAX_RETAINED_STEPS,
                "retained_artifacts": retained_artifacts,
                "deduplicated_by_factor_state_sha256": True,
                "portable_guided_source_binding_sha256": (
                    self.portable_guided_source_binding_sha256
                ),
                "historical_guided_source_binding_sha256": (
                    self.historical_guided_source_binding.binding_sha256
                ),
                "runtime_identity_excluded_from_portable_content": sorted(
                    _RUNTIME_SOURCE_BINDING_FIELDS
                ),
                "historical_durable_checkpoint_sha_required": False,
                "replay_durable_checkpoint_sha_self_authenticated": True,
                "all_expected_callbacks_seen": True,
                "terminal_rows_match_callbacks": True,
                "contains_parent_state": False,
                "contains_native_child_state": False,
                "contains_optimizer_state": False,
                "contains_model_checkpoint": False,
                "publication": "atomic-sibling-directory-rename",
            }
            payload["manifest_sha256"] = canonical_sha256(payload)
            manifest_path = self._staging_root / _RESEARCH_MANIFEST_FILENAME
            _atomic_write_json(manifest_path, payload)
            manifest_file_sha = file_sha256(manifest_path)
            _fsync_directory(self._retained_root)
            _fsync_directory(self._staging_root)
            if self.output_root.exists():
                raise GuidedCheckpointRecoveryPublicationError(
                    "Recovery output appeared before atomic publication."
                )
            os.rename(self._staging_root, self.output_root)
            _fsync_directory(self.output_root.parent)
            self._staging_root = None
            self._state = "finalized"
            return GuidedCheckpointRecoveryPublication(
                output_root=self.output_root,
                manifest_path=self.output_root / _RESEARCH_MANIFEST_FILENAME,
                manifest_sha256=str(payload["manifest_sha256"]),
                manifest_file_sha256=manifest_file_sha,
                retained_steps=self.retained_steps,
                retained_factor_state_sha256s=tuple(
                    value["factor_state_sha256"] for value in retained_artifacts
                ),
            )
        except BaseException:
            self._fail()
            raise


def rebind_recovered_v3_factor_payload(
    payload: Mapping[str, Any],
    runtime_guided_source_binding: GuidedFactorSourceBinding,
) -> dict[str, Any]:
    """Authenticate and rebind a retained v3 pack for a new live process.

    The stored pack remains byte-for-byte in its historical legacy hash domain.
    This function first authenticates that pack, proves all content-bearing
    source fields equal the caller's separately authenticated live binding, and
    only then returns a cloned payload whose legacy digest is valid for that
    runtime installation.  Tensor bytes are never rewritten.
    """

    if not isinstance(payload, Mapping):
        raise GuidedCheckpointRecoveryAuthenticationError(
            "Recovered v3 factor payload must be a mapping."
        )
    source_binding = _source_binding(
        payload.get("guided_source_binding"),
        "retained historical guided source binding",
    )
    expected_source_sha = _sha256(
        payload.get("factor_state_sha256"),
        "retained historical factor state",
    )
    GuidedCheckpointRecoverySink._validate_v3_factor_payload(
        payload,
        expected_factor_state_sha256=expected_source_sha,
    )
    runtime_binding = _source_binding(
        runtime_guided_source_binding,
        "runtime guided source binding",
    )
    if canonical_json_bytes(portable_guided_source_binding(source_binding)) != (
        canonical_json_bytes(portable_guided_source_binding(runtime_binding))
    ):
        raise GuidedCheckpointRecoveryAuthenticationError(
            "Retained pack source content differs from the live runtime sources."
        )
    portable_before = portable_v3_factor_state_sha256(payload, source_binding)
    rebound = _rebind_v3_factor_payload(payload, runtime_binding)
    GuidedCheckpointRecoverySink._validate_v3_factor_payload(
        rebound,
        expected_factor_state_sha256=str(rebound["factor_state_sha256"]),
    )
    if portable_v3_factor_state_sha256(rebound, runtime_binding) != portable_before:
        raise GuidedCheckpointRecoveryAuthenticationError(
            "Runtime rebinding changed portable factor content identity."
        )
    return rebound


__all__ = [
    "GUIDED_CHECKPOINT_CANDIDATE_SCHEMA",
    "GUIDED_CHECKPOINT_PORTABLE_FACTOR_IDENTITY_SCHEMA",
    "GUIDED_CHECKPOINT_RECOVERY_SCHEMA",
    "GUIDED_CHECKPOINT_RESEARCH_MANIFEST_SCHEMA",
    "GuidedCheckpointRecoveryAuthenticationError",
    "GuidedCheckpointRecoveryError",
    "GuidedCheckpointRecoveryOrderError",
    "GuidedCheckpointRecoveryPublication",
    "GuidedCheckpointRecoveryPublicationError",
    "GuidedCheckpointRecoverySink",
    "GuidedCheckpointRecoveryStateError",
    "HistoricalCheckpointExpectation",
    "canonical_json_bytes",
    "canonical_sha256",
    "file_sha256",
    "portable_guided_source_binding",
    "portable_guided_source_binding_sha256",
    "portable_v3_factor_state_sha256",
    "rebind_recovered_v3_factor_payload",
]
