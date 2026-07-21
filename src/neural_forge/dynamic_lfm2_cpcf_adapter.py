"""Catalog integration for the cached predictive circuit fold (CPCF).

The CPCF compiler is intentionally a model-level fold.  This adapter exposes
it through the same target-catalog, evidence, transaction, behavior-gate, and
fresh-reload contracts as every other production dynamic-fold lane.

Large parent-side phase tensors are authenticated while they are materialized
and rehashed at both sides of the compiler call.  A compile-scoped Torch
dispatch firewall additionally rejects every attempted write to the evidence,
phase inputs, tied interface, parent checkpoint, and live blank.  The compiler
receives only a disposable staging copy of the blank; an authenticated
``copy_`` installation is the sole mutation path into the live child.
"""

from __future__ import annotations

import hashlib
import json
import marshal
import math
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from types import CodeType, FunctionType, MappingProxyType, ModuleType
from typing import TYPE_CHECKING, Any

from . import dynamic_replay_scheduler as _dynamic_replay_scheduler_module
from . import hf_lfm2_cpcf_candidate_replay as _candidate_replay_module
from . import hf_lfm2_cpcf_resume_artifact as _resume_artifact_module
from . import hf_lfm2_cpcf_v5_admission_proof as _v5_admission_proof_module
from . import hf_lfm2_cpcf_v5_model_compiler as _v5_model_compiler_module
from . import hf_lfm2_cpcf_v5_resume as _v5_resume_module
from .dynamic_behavior_gate import (
    BehaviorGateBinding,
    DynamicBehaviorGateError,
    resolve_behavior_gate_binding,
)
from .dynamic_circuit_ir import JsonScalar, ModelCircuitGraph, canonical_sha256
from .dynamic_fold_artifact import _callable_fingerprints
from .dynamic_fold_compiler import DynamicFoldSystemStack
from .dynamic_fold_executor import (
    FoldDecision,
    FoldEvaluationContext,
    FoldEvidence,
    FoldExecutionContext,
    FoldOperatorRegistry,
    FoldTransaction,
    fingerprint_model_runtime_state,
    fingerprint_model_state,
)
from .dynamic_fold_planner import DynamicFoldPlan, FoldAction
from .dynamic_fold_snapshot import DynamicFoldSnapshotConfig
from .dynamic_lfm2_adapter import (
    Lfm2DenseBudgetedSelectedTarget,
    Lfm2TargetSynthesisAdapter,
    encode_lfm2_layer_types,
    encode_lfm2_source_spans,
    validate_lfm2_dense_budgeted_child,
)
from .dynamic_lfm2_cpcf_reviewed_gate import (
    ORDERED_PHASE_BINDING_SCHEMA,
    REVIEWED_EVIDENCE_SCHEMA,
    Lfm2CpcfReviewedPromotionEvidence,
    Lfm2CpcfReviewedPromotionError,
    Lfm2CpcfReviewedPromotionGate,
    authenticate_lfm2_cpcf_reviewed_promotion_gate,
    fingerprint_cpcf_candidate_state,
)
from .dynamic_target_synthesis import (
    DynamicTargetSynthesisError,
    DynamicTargetSynthesisResult,
    verify_compiler_authenticated_target,
)
from .hf_lfm2_cpcf_model_compiler import Lfm2CpcfTargetPhase
from .hf_lfm2_cpcf_candidate_replay import (
    LFM2_CPCF_CANDIDATE_REPLAY_BACKEND_SCHEMA,
    AuthenticatedTwoGpuLfm2CpcfCandidateReplayEvaluator,
    lfm2_cpcf_cpu_parity_worker,
    lfm2_cpcf_cuda_replay_worker,
    replay_lfm2_cpcf_candidate_work_payload,
)
from .hf_lfm2_circuit_trace_cache import Lfm2CircuitTraceCache
from .hf_lfm2_parent_relative_manual_gate import (
    PROMOTION_ELIGIBLE_SUITE_CONCLUSION,
)
from .hf_lfm2_cpcf_model_fold import (
    LFM2_CPCF_MODEL_FOLD_SCHEMA,
    Lfm2CpcfCompiledModelFold,
    Lfm2CpcfCompletedLayerReceipt,
    Lfm2CpcfModelFoldPolicy,
    compile_lfm2_cpcf_model_fold,
)
from .hf_lfm2_cpcf_selective_evidence import (
    Lfm2CpcfSelectiveEvidenceProvider,
    build_lfm2_cpcf_selective_evidence_provider,
)
from .hf_lfm2_cpcf_short_conv_v5_numerical_adapter import (
    Lfm2CpcfShortConvV5NumericalFrontierAdapter,
)
from .hf_lfm2_cpcf_v5_model_compiler import (
    LFM2_CPCF_V5_MODEL_FOLD_SCHEMA,
    LFM2_CPCF_V5_PHYSICAL_RSS_SCHEMA,
    Lfm2CpcfV5EndpointFirstCandidateSelector,
    Lfm2CpcfV5RuntimePolicy,
    _prove_real_v5_numerical_layer,
    _real_v5_numerical_adapter_contract,
    _optional_runtime_contract_sha256,
    compile_lfm2_cpcf_v5_selective_model_fold,
    lfm2_cpcf_v5_compiler_implementation,
)
from .hf_lfm2_cpcf_v5_admission_proof import (
    authenticate_lfm2_cpcf_v5_admission_chain_proof,
)
from .hf_lfm2_cpcf_v5_resume import (
    LFM2_CPCF_V5_RESUME_SCHEMA,
    Lfm2CpcfV5ResumeCoordinator,
    Lfm2CpcfV5ResumeState,
    build_lfm2_cpcf_v5_physical_rss_policy_sha256,
    lfm2_cpcf_v5_resume_implementation,
)
from .hf_lfm2_cpcf_resume_artifact import (
    Lfm2CpcfArtifactReference,
    build_lfm2_cpcf_target_topology_snapshot,
)
from . import hf_lfm2_cpcf_model_fold as _lfm2_cpcf_model_fold_module
from .hf_lfm2_dense_budget_model_fold import DENSE_BUDGET_MODEL_METHOD
from .hf_lfm2_tokenizer_bridge_trace_adapter import (
    Lfm2StrictTokenizerBridgeTraceAdapter,
)
from .dynamic_replay_scheduler import (
    AuthenticatedPinnedHostBufferFactory,
    AuthenticatedReplayStateProbes,
    DYNAMIC_REPLAY_EXECUTION_SCHEMA,
    DeterministicReplayScheduler,
    PinnedHostDoubleBuffer,
    ReplaySchedulerPolicy,
    capture_cuda_replay_device_receipt,
)

if TYPE_CHECKING:
    from .dynamic_lfm2_cpcf_resources import (
        Lfm2CpcfCachePhaseCoverageBinding,
    )


LFM2_CPCF_STACK_ADAPTER_ID = "neural_forge.lfm2.cpcf_catalog"
LFM2_CPCF_CATALOG_SCHEMA = "neural-forge-lfm2-cpcf-catalog-stack-v1"
LFM2_CPCF_BLANK_REPORT_SCHEMA = "neural-forge-lfm2-cpcf-catalog-blank-v1"
LFM2_CPCF_PARENT_TENSOR_SEAL_SCHEMA = "neural-forge-lfm2-cpcf-parent-tensor-seal-v1"
LFM2_CPCF_PARAMETER_PATH_WRITE_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-cpcf-parameter-path-write-receipt-v1"
)
LFM2_CPCF_STAGED_INSTALL_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-cpcf-staged-install-receipt-v1"
)
LFM2_CPCF_COMPILE_WRITE_GUARD_SCHEMA = "neural-forge-lfm2-cpcf-compile-write-guard-v1"
LFM2_CPCF_CATALOG_COMPILE_REPORT_SCHEMA = (
    "neural-forge-lfm2-cpcf-catalog-compile-report-v1"
)
LFM2_CPCF_RESOURCE_SCHEMA = "neural-forge-lfm2-cpcf-catalog-resource-v1"
LFM2_CPCF_V5_BACKEND_SCHEMA = "neural-forge-lfm2-cpcf-v5-selective-catalog-backend-v1"
LFM2_CPCF_V5_RESOURCE_SCHEMA = "neural-forge-lfm2-cpcf-v5-selective-catalog-resource-v1"
LFM2_CPCF_V5_EVIDENCE_SCHEMA = "neural-forge-lfm2-cpcf-v5-selective-catalog-evidence-v1"
LFM2_CPCF_V5_GPU_REPLAY_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-authenticated-two-gpu-replay-v1"
)
LFM2_CPCF_V5_GPU_EXECUTION_BINDING_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-gpu-execution-binding-v1"
)
LFM2_CPCF_V5_GPU_PARITY_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-cpcf-candidate-gpu-parity-v1"
)
LFM2_CPCF_EXTERNAL_GATE_SCHEMA = "neural-forge-lfm2-cpcf-external-intelligence-gate-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_V5_GPU_PARITY_SOURCE_FILENAMES = frozenset(
    {
        "dynamic_replay_scheduler.py",
        "hf_lfm2_cpcf_candidate_replay.py",
        "verify_lfm2_cpcf_candidate_replay_gpu_parity.py",
    }
)
_V5_GPU_PARITY_RECEIPT_FIELDS = frozenset(
    {
        "backend_report",
        "backward_calls",
        "candidate_count",
        "candidate_state_sha256_equal",
        "candidate_work_items",
        "case_count",
        "closed_form_solve_math_changed",
        "devices",
        "direct_cpu_report_sha256",
        "maximum_allowed_metric_delta",
        "maximum_metric_delta",
        "metric_deltas",
        "optimizer_steps",
        "parameters_pruned",
        "parameters_quantized",
        "receipt_sha256",
        "scheduled_cuda_report_sha256",
        "schema",
        "solver_policy_sha256",
        "source_sha256",
        "status",
        "target_dtype",
        "training_steps",
    }
)

# Keep the exact imported compiler object independently of the public module
# name below. An ordinary monkeypatch of the adapter dispatch name must never
# replace the production compiler after a backend has been authenticated.
_AUTHENTIC_V5_COMPILER_CALLABLE = compile_lfm2_cpcf_v5_selective_model_fold
_AUTHENTIC_V5_COMPILER_CODE_SHA256 = hashlib.sha256(
    marshal.dumps(_AUTHENTIC_V5_COMPILER_CALLABLE.__code__)
).hexdigest()
_AUTHENTIC_V5_COMPILER_SOURCE_PATH = Path(
    _AUTHENTIC_V5_COMPILER_CALLABLE.__code__.co_filename
).resolve()
_AUTHENTIC_V5_COMPILER_SOURCE_SHA256 = hashlib.sha256(
    _AUTHENTIC_V5_COMPILER_SOURCE_PATH.read_bytes()
).hexdigest()
_AUTHENTIC_TWO_GPU_EVALUATOR_CLASS = AuthenticatedTwoGpuLfm2CpcfCandidateReplayEvaluator
_AUTHENTIC_REPLAY_SCHEDULER_CLASS = DeterministicReplayScheduler
_AUTHENTIC_REPLAY_SCHEDULER_POLICY_CLASS = ReplaySchedulerPolicy
_AUTHENTIC_REPLAY_STATE_PROBES_CLASS = AuthenticatedReplayStateProbes
_AUTHENTIC_PINNED_HOST_FACTORY_CLASS = AuthenticatedPinnedHostBufferFactory
_AUTHENTIC_PINNED_HOST_BUFFER_CLASS = PinnedHostDoubleBuffer
_AUTHENTIC_CUDA_REPLAY_WORKER = lfm2_cpcf_cuda_replay_worker
_AUTHENTIC_CPU_PARITY_WORKER = lfm2_cpcf_cpu_parity_worker
_AUTHENTIC_CANDIDATE_REPLAY_PAYLOAD = replay_lfm2_cpcf_candidate_work_payload
_AUTHENTIC_DEVICE_RECEIPT_PROBE = capture_cuda_replay_device_receipt
_V5_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_V5_GPU_PARITY_FIXED_MAXIMUM_METRIC_DELTA = 0.05


class Lfm2CpcfCatalogError(RuntimeError):
    """The CPCF catalog stack is incomplete or unauthenticated."""


@dataclass(frozen=True, slots=True)
class _V5GpuParityAuthority:
    """Source-controlled approval for one exact production parity artifact."""

    repository_relative_path: str
    file_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        relative = Path(self.repository_relative_path)
        if (
            not self.repository_relative_path
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != self.repository_relative_path
        ):
            raise Lfm2CpcfCatalogError(
                "Approved V5 GPU parity path must be canonical and repository-relative."
            )
        _digest(self.file_sha256, "approved V5 GPU parity file SHA-256")
        _digest(
            self.receipt_sha256,
            "approved V5 GPU parity receipt SHA-256",
        )


# Intentionally empty. After the next real verifier run on two RTX 3090s,
# replace only this value with one _V5GpuParityAuthority carrying the exact
# canonical repository-relative report path plus its file and receipt hashes.
# Production must remain fail-closed until that explicit source review occurs.
_PRODUCTION_V5_GPU_PARITY_AUTHORITY: _V5GpuParityAuthority | None = None


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for LFM2 CPCF folding.") from exc
    return torch


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2CpcfCatalogError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2CpcfCatalogError(f"{name} must be nonempty text.")
    return value.strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _v5_stable_code_sha256(code: Any) -> str:
    """Hash code structurally, independent of CPython string interning state."""

    digest = hashlib.sha256()

    def update(value: Any) -> None:
        nonlocal digest
        if isinstance(value, CodeType):
            digest.update(b"code{")
            for field_value in (
                value.co_argcount,
                value.co_posonlyargcount,
                value.co_kwonlyargcount,
                value.co_nlocals,
                value.co_stacksize,
                value.co_flags,
                value.co_code,
                value.co_consts,
                value.co_names,
                value.co_varnames,
                value.co_filename,
                value.co_name,
                value.co_qualname,
                value.co_firstlineno,
                value.co_linetable,
                value.co_exceptiontable,
                value.co_freevars,
                value.co_cellvars,
            ):
                update(field_value)
            digest.update(b"}")
            return
        if value is None:
            digest.update(b"none;")
            return
        if value is Ellipsis:
            digest.update(b"ellipsis;")
            return
        if isinstance(value, bool):
            digest.update(b"bool:1;" if value else b"bool:0;")
            return
        if isinstance(value, int):
            encoded = str(value).encode("ascii")
            digest.update(b"int:" + str(len(encoded)).encode("ascii") + b":")
            digest.update(encoded)
            return
        if isinstance(value, float):
            encoded = value.hex().encode("ascii")
            digest.update(b"float:" + str(len(encoded)).encode("ascii") + b":")
            digest.update(encoded)
            return
        if isinstance(value, complex):
            digest.update(b"complex{")
            update(value.real)
            update(value.imag)
            digest.update(b"}")
            return
        if isinstance(value, bytes):
            digest.update(b"bytes:" + str(len(value)).encode("ascii") + b":")
            digest.update(value)
            return
        if isinstance(value, str):
            encoded = value.encode("utf-8")
            digest.update(b"str:" + str(len(encoded)).encode("ascii") + b":")
            digest.update(encoded)
            return
        if isinstance(value, tuple):
            digest.update(b"tuple:" + str(len(value)).encode("ascii") + b"{")
            for child in value:
                update(child)
            digest.update(b"}")
            return
        if isinstance(value, frozenset):
            child_hashes = []
            for child in value:
                prior_digest = digest
                try:
                    digest_copy = hashlib.sha256()
                    digest = digest_copy
                    update(child)
                    child_hashes.append(digest_copy.digest())
                finally:
                    digest = prior_digest
            digest.update(b"frozenset:" + str(len(child_hashes)).encode("ascii") + b"{")
            for child_hash in sorted(child_hashes):
                digest.update(child_hash)
            digest.update(b"}")
            return
        raise RuntimeError(
            f"Unsupported V5 Python code constant type: {type(value)!r}."
        )

    update(code)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _V5RuntimeCallablePin:
    label: str
    callable_object: Callable[..., Any]
    code_object: CodeType
    code_sha256: str
    defaults: tuple[Any, ...] | None
    kwdefaults: tuple[tuple[str, Any], ...] | None
    closure_objects: tuple[Any, ...] | None
    source_path: Path
    source_sha256: str

    @classmethod
    def create(
        cls,
        label: str,
        callable_object: Callable[..., Any],
    ) -> _V5RuntimeCallablePin:
        code = getattr(callable_object, "__code__", None)
        if code is None:
            raise RuntimeError(
                f"V5 runtime callable {label!r} has no Python code object."
            )
        source_path = Path(code.co_filename).resolve()
        if not source_path.is_file() or source_path.is_symlink():
            raise RuntimeError(
                f"V5 runtime callable {label!r} has no safe source file."
            )
        return cls(
            label=label,
            callable_object=callable_object,
            code_object=code,
            code_sha256=_v5_stable_code_sha256(code),
            defaults=(
                None
                if callable_object.__defaults__ is None
                else tuple(callable_object.__defaults__)
            ),
            kwdefaults=(
                None
                if callable_object.__kwdefaults__ is None
                else tuple(sorted(callable_object.__kwdefaults__.items()))
            ),
            closure_objects=(
                None
                if callable_object.__closure__ is None
                else tuple(cell.cell_contents for cell in callable_object.__closure__)
            ),
            source_path=source_path,
            source_sha256=_file_sha256(source_path),
        )

    def authenticate(self) -> Mapping[str, Any]:
        code = getattr(self.callable_object, "__code__", None)
        defaults = (
            None
            if self.callable_object.__defaults__ is None
            else tuple(self.callable_object.__defaults__)
        )
        kwdefaults = (
            None
            if self.callable_object.__kwdefaults__ is None
            else tuple(sorted(self.callable_object.__kwdefaults__.items()))
        )
        closures = (
            None
            if self.callable_object.__closure__ is None
            else tuple(cell.cell_contents for cell in self.callable_object.__closure__)
        )
        if (
            code is not self.code_object
            or not self.source_path.is_file()
            or self.source_path.is_symlink()
            or _v5_stable_code_sha256(code) != self.code_sha256
            or defaults != self.defaults
            or kwdefaults != self.kwdefaults
            or (closures is None) != (self.closure_objects is None)
            or (
                closures is not None
                and self.closure_objects is not None
                and (
                    len(closures) != len(self.closure_objects)
                    or any(
                        observed is not expected
                        for observed, expected in zip(
                            closures,
                            self.closure_objects,
                            strict=True,
                        )
                    )
                )
            )
            or _file_sha256(self.source_path) != self.source_sha256
        ):
            raise Lfm2CpcfCatalogError(
                f"V5 runtime callable {self.label!r} changed after import."
            )
        return MappingProxyType(
            {
                "label": self.label,
                "module": self.callable_object.__module__,
                "qualname": self.callable_object.__qualname__,
                "code_sha256": self.code_sha256,
                "source_path": str(self.source_path),
                "source_sha256": self.source_sha256,
            }
        )


def _v5_class_callable_descriptor(
    value: Any,
) -> tuple[str, tuple[tuple[str, Callable[..., Any]], ...]] | None:
    """Return the exact Python callables carried by one class descriptor."""

    if isinstance(value, FunctionType):
        return ("function", (("call", value),))
    if isinstance(value, staticmethod):
        return ("staticmethod", (("call", value.__func__),))
    if isinstance(value, classmethod):
        return ("classmethod", (("call", value.__func__),))
    if isinstance(value, property):
        accessors = tuple(
            (role, function)
            for role, function in (
                ("get", value.fget),
                ("set", value.fset),
                ("delete", value.fdel),
            )
            if function is not None
        )
        if accessors:
            return ("property", accessors)
    return None


@dataclass(frozen=True, slots=True)
class _V5RuntimeClassCallablePin:
    """Immutable identity/code pin for one callable class descriptor."""

    mro_index: int
    owner: type
    name: str
    descriptor_kind: str
    descriptor_object: Any
    accessor_roles: tuple[str, ...]
    accessor_objects: tuple[Callable[..., Any], ...]
    accessor_code_objects: tuple[CodeType, ...]
    accessor_code_sha256: tuple[str, ...]
    accessor_defaults: tuple[tuple[Any, ...] | None, ...]
    accessor_kwdefaults: tuple[
        tuple[tuple[str, Any], ...] | None,
        ...,
    ]
    accessor_closure_objects: tuple[tuple[Any, ...] | None, ...]

    @classmethod
    def create(
        cls,
        *,
        mro_index: int,
        owner: type,
        name: str,
        descriptor_object: Any,
        descriptor_kind: str,
        accessors: tuple[tuple[str, Callable[..., Any]], ...],
    ) -> _V5RuntimeClassCallablePin:
        codes = tuple(getattr(function, "__code__", None) for _, function in accessors)
        if any(code is None for code in codes):
            raise RuntimeError(
                f"V5 runtime class callable {owner.__qualname__}.{name} "
                "has no Python code object."
            )
        return cls(
            mro_index=mro_index,
            owner=owner,
            name=name,
            descriptor_kind=descriptor_kind,
            descriptor_object=descriptor_object,
            accessor_roles=tuple(role for role, _ in accessors),
            accessor_objects=tuple(function for _, function in accessors),
            accessor_code_objects=tuple(codes),
            accessor_code_sha256=tuple(_v5_stable_code_sha256(code) for code in codes),
            accessor_defaults=tuple(
                None if function.__defaults__ is None else tuple(function.__defaults__)
                for _, function in accessors
            ),
            accessor_kwdefaults=tuple(
                None
                if function.__kwdefaults__ is None
                else tuple(sorted(function.__kwdefaults__.items()))
                for _, function in accessors
            ),
            accessor_closure_objects=tuple(
                None
                if function.__closure__ is None
                else tuple(cell.cell_contents for cell in function.__closure__)
                for _, function in accessors
            ),
        )

    def authenticate(
        self,
        *,
        current_descriptor: Any,
        current_kind: str,
        current_accessors: tuple[tuple[str, Callable[..., Any]], ...],
    ) -> Mapping[str, Any]:
        current_roles = tuple(role for role, _ in current_accessors)
        current_objects = tuple(function for _, function in current_accessors)
        current_codes = tuple(
            getattr(function, "__code__", None) for function in current_objects
        )
        current_defaults = tuple(
            None if function.__defaults__ is None else tuple(function.__defaults__)
            for function in current_objects
        )
        current_kwdefaults = tuple(
            None
            if function.__kwdefaults__ is None
            else tuple(sorted(function.__kwdefaults__.items()))
            for function in current_objects
        )
        current_closures = tuple(
            None
            if function.__closure__ is None
            else tuple(cell.cell_contents for cell in function.__closure__)
            for function in current_objects
        )
        if (
            current_descriptor is not self.descriptor_object
            or current_kind != self.descriptor_kind
            or current_roles != self.accessor_roles
            or len(current_objects) != len(self.accessor_objects)
            or any(
                observed is not expected
                for observed, expected in zip(
                    current_objects,
                    self.accessor_objects,
                    strict=True,
                )
            )
            or any(code is None for code in current_codes)
            or any(
                observed is not expected
                for observed, expected in zip(
                    current_codes,
                    self.accessor_code_objects,
                    strict=True,
                )
            )
            or tuple(_v5_stable_code_sha256(code) for code in current_codes)
            != self.accessor_code_sha256
            or current_defaults != self.accessor_defaults
            or current_kwdefaults != self.accessor_kwdefaults
            or len(current_closures) != len(self.accessor_closure_objects)
            or any(
                (observed is None and expected is not None)
                or (observed is not None and expected is None)
                or (
                    observed is not None
                    and expected is not None
                    and (
                        len(observed) != len(expected)
                        or any(
                            observed_value is not expected_value
                            for observed_value, expected_value in zip(
                                observed,
                                expected,
                                strict=True,
                            )
                        )
                    )
                )
                for observed, expected in zip(
                    current_closures,
                    self.accessor_closure_objects,
                    strict=True,
                )
            )
        ):
            raise Lfm2CpcfCatalogError(
                "V5 runtime class callable manifest changed after import: "
                f"{self.owner.__qualname__}.{self.name}."
            )
        return MappingProxyType(
            {
                "mro_index": self.mro_index,
                "owner_module": self.owner.__module__,
                "owner_qualname": self.owner.__qualname__,
                "name": self.name,
                "descriptor_kind": self.descriptor_kind,
                "accessors": [
                    {
                        "role": role,
                        "module": function.__module__,
                        "qualname": function.__qualname__,
                        "code_sha256": code_sha256,
                        "defaults_authenticated": True,
                        "closure_authenticated": True,
                    }
                    for role, function, code_sha256 in zip(
                        self.accessor_roles,
                        self.accessor_objects,
                        self.accessor_code_sha256,
                        strict=True,
                    )
                ],
            }
        )


@dataclass(frozen=True, slots=True)
class _V5RuntimeClassPin:
    """Complete callable-descriptor and source-byte manifest for one class."""

    label: str
    class_object: type
    exact_mro: tuple[type, ...]
    callable_pins: tuple[_V5RuntimeClassCallablePin, ...]
    source_paths: tuple[Path, ...]
    source_sha256: tuple[str, ...]

    @classmethod
    def create(
        cls,
        label: str,
        class_object: type,
    ) -> _V5RuntimeClassPin:
        callable_pins: list[_V5RuntimeClassCallablePin] = []
        source_paths: set[Path] = set()
        exact_mro = tuple(class_object.__mro__)
        for mro_index, owner in enumerate(exact_mro):
            if owner is object or owner.__module__ != class_object.__module__:
                continue
            for name, descriptor in vars(owner).items():
                parsed = _v5_class_callable_descriptor(descriptor)
                if parsed is None:
                    continue
                descriptor_kind, accessors = parsed
                callable_pins.append(
                    _V5RuntimeClassCallablePin.create(
                        mro_index=mro_index,
                        owner=owner,
                        name=name,
                        descriptor_object=descriptor,
                        descriptor_kind=descriptor_kind,
                        accessors=accessors,
                    )
                )
                for _, function in accessors:
                    code = function.__code__
                    candidate = Path(code.co_filename).resolve()
                    if candidate.is_file() and not candidate.is_symlink():
                        source_paths.add(candidate)
        module = sys.modules.get(class_object.__module__)
        module_file = getattr(module, "__file__", None)
        if isinstance(module_file, str):
            candidate = Path(module_file).resolve()
            if candidate.is_file() and not candidate.is_symlink():
                source_paths.add(candidate)
        ordered_paths = tuple(sorted(source_paths, key=lambda path: str(path)))
        if not ordered_paths:
            raise RuntimeError(f"V5 runtime class {label!r} has no safe source file.")
        return cls(
            label=label,
            class_object=class_object,
            exact_mro=exact_mro,
            callable_pins=tuple(callable_pins),
            source_paths=ordered_paths,
            source_sha256=tuple(_file_sha256(path) for path in ordered_paths),
        )

    def authenticate(self) -> Mapping[str, Any]:
        if tuple(self.class_object.__mro__) != self.exact_mro:
            raise Lfm2CpcfCatalogError(
                f"V5 runtime class {self.label!r} MRO changed after import."
            )
        current: dict[
            tuple[int, str],
            tuple[
                Any,
                str,
                tuple[tuple[str, Callable[..., Any]], ...],
            ],
        ] = {}
        for mro_index, owner in enumerate(self.exact_mro):
            if owner is object or owner.__module__ != self.class_object.__module__:
                continue
            for name, descriptor in vars(owner).items():
                parsed = _v5_class_callable_descriptor(descriptor)
                if parsed is None:
                    continue
                descriptor_kind, accessors = parsed
                current[(mro_index, name)] = (
                    descriptor,
                    descriptor_kind,
                    accessors,
                )
        expected_keys = {(pin.mro_index, pin.name) for pin in self.callable_pins}
        if set(current) != expected_keys:
            raise Lfm2CpcfCatalogError(
                f"V5 runtime class {self.label!r} callable name set "
                "changed after import."
            )
        authenticated_callables = []
        for pin in self.callable_pins:
            descriptor, descriptor_kind, accessors = current[(pin.mro_index, pin.name)]
            authenticated_callables.append(
                dict(
                    pin.authenticate(
                        current_descriptor=descriptor,
                        current_kind=descriptor_kind,
                        current_accessors=accessors,
                    )
                )
            )
        if any(
            not path.is_file()
            or path.is_symlink()
            or _file_sha256(path) != expected_sha256
            for path, expected_sha256 in zip(
                self.source_paths,
                self.source_sha256,
                strict=True,
            )
        ):
            raise Lfm2CpcfCatalogError(
                f"V5 runtime class {self.label!r} source bytes changed after import."
            )
        return MappingProxyType(
            {
                "label": self.label,
                "module": self.class_object.__module__,
                "qualname": self.class_object.__qualname__,
                "mro": [
                    f"{owner.__module__}.{owner.__qualname__}"
                    for owner in self.exact_mro
                ],
                "callables": authenticated_callables,
                "sources": [
                    {
                        "path": str(path),
                        "sha256": sha256,
                    }
                    for path, sha256 in zip(
                        self.source_paths,
                        self.source_sha256,
                        strict=True,
                    )
                ],
            }
        )


@dataclass(frozen=True, slots=True)
class _V5RuntimeModulePin:
    """Complete non-dunder namespace manifest for one runtime module."""

    label: str
    module_object: ModuleType
    namespace_names: tuple[str, ...]
    namespace_objects: tuple[Any, ...]
    function_pins: tuple[_V5RuntimeCallablePin, ...]
    class_pins: tuple[_V5RuntimeClassPin, ...]
    helper_binding_names: tuple[str, ...]
    source_path: Path
    source_sha256: str

    @classmethod
    def create(
        cls,
        label: str,
        module_object: ModuleType,
    ) -> _V5RuntimeModulePin:
        module_file = getattr(module_object, "__file__", None)
        if not isinstance(module_file, str):
            raise RuntimeError(f"V5 runtime module {label!r} has no source file.")
        source_path = Path(module_file).resolve()
        if not source_path.is_file() or source_path.is_symlink():
            raise RuntimeError(f"V5 runtime module {label!r} has no safe source file.")
        namespace_names = tuple(
            sorted(name for name in vars(module_object) if not name.startswith("__"))
        )
        namespace_objects = tuple(vars(module_object)[name] for name in namespace_names)
        function_pins = tuple(
            _V5RuntimeCallablePin.create(
                f"{label}.{name}",
                value,
            )
            for name, value in zip(
                namespace_names,
                namespace_objects,
                strict=True,
            )
            if isinstance(value, FunctionType)
        )
        class_pins = tuple(
            _V5RuntimeClassPin.create(
                f"{label}.{name}",
                value,
            )
            for name, value in zip(
                namespace_names,
                namespace_objects,
                strict=True,
            )
            if isinstance(value, type) and value.__module__ == module_object.__name__
        )
        helper_binding_names = tuple(
            name
            for name, value in zip(
                namespace_names,
                namespace_objects,
                strict=True,
            )
            if isinstance(value, FunctionType)
            or (isinstance(value, type) and value.__module__ == module_object.__name__)
        )
        return cls(
            label=label,
            module_object=module_object,
            namespace_names=namespace_names,
            namespace_objects=namespace_objects,
            function_pins=function_pins,
            class_pins=class_pins,
            helper_binding_names=helper_binding_names,
            source_path=source_path,
            source_sha256=_file_sha256(source_path),
        )

    def authenticate(self) -> Mapping[str, Any]:
        current_names = tuple(
            sorted(
                name for name in vars(self.module_object) if not name.startswith("__")
            )
        )
        if current_names != self.namespace_names:
            raise Lfm2CpcfCatalogError(
                f"V5 runtime module {self.label!r} namespace name set "
                "changed after import."
            )
        for name, expected in zip(
            self.namespace_names,
            self.namespace_objects,
            strict=True,
        ):
            if vars(self.module_object).get(name) is not expected:
                raise Lfm2CpcfCatalogError(
                    f"V5 runtime module global {self.label}.{name} "
                    "changed after import."
                )
        if (
            not self.source_path.is_file()
            or self.source_path.is_symlink()
            or _file_sha256(self.source_path) != self.source_sha256
        ):
            raise Lfm2CpcfCatalogError(
                f"V5 runtime module {self.label!r} source bytes changed after import."
            )
        authenticated_functions = [
            dict(pin.authenticate()) for pin in self.function_pins
        ]
        authenticated_classes = [dict(pin.authenticate()) for pin in self.class_pins]
        return MappingProxyType(
            {
                "label": self.label,
                "module": self.module_object.__name__,
                "namespace_names": list(self.namespace_names),
                "source_path": str(self.source_path),
                "source_sha256": self.source_sha256,
                "functions": authenticated_functions,
                "classes": authenticated_classes,
            }
        )


@dataclass(frozen=True, slots=True)
class _V5RuntimeReachableGlobalPin:
    """One exact global binding reached by authenticated runtime bytecode."""

    owner_module: str
    owner_globals: dict[str, Any]
    name: str
    binding_object: Any
    callable_pin: _V5RuntimeCallablePin | None
    class_pin: _V5RuntimeClassPin | None

    @classmethod
    def create(
        cls,
        *,
        owner_module: str,
        owner_globals: dict[str, Any],
        name: str,
        binding_object: Any,
    ) -> _V5RuntimeReachableGlobalPin:
        callable_pin = (
            _V5RuntimeCallablePin.create(
                f"reachable.{owner_module}.{name}",
                binding_object,
            )
            if isinstance(binding_object, FunctionType)
            else None
        )
        class_pin = (
            _V5RuntimeClassPin.create(
                f"reachable.{owner_module}.{name}",
                binding_object,
            )
            if isinstance(binding_object, type)
            and binding_object.__module__.startswith("neural_forge.")
            else None
        )
        return cls(
            owner_module=owner_module,
            owner_globals=owner_globals,
            name=name,
            binding_object=binding_object,
            callable_pin=callable_pin,
            class_pin=class_pin,
        )

    def authenticate(self) -> Mapping[str, Any]:
        if self.owner_globals.get(self.name) is not self.binding_object:
            raise Lfm2CpcfCatalogError(
                "V5 reachable runtime global "
                f"{self.owner_module}.{self.name} changed after import."
            )
        callable_report = (
            None
            if self.callable_pin is None
            else dict(self.callable_pin.authenticate())
        )
        class_report = (
            None if self.class_pin is None else dict(self.class_pin.authenticate())
        )
        return MappingProxyType(
            {
                "owner_module": self.owner_module,
                "name": self.name,
                "binding_module": getattr(
                    self.binding_object,
                    "__module__",
                    type(self.binding_object).__module__,
                ),
                "binding_qualname": getattr(
                    self.binding_object,
                    "__qualname__",
                    type(self.binding_object).__qualname__,
                ),
                "binding_type": (
                    f"{type(self.binding_object).__module__}."
                    f"{type(self.binding_object).__qualname__}"
                ),
                "callable": callable_report,
                "class": class_report,
            }
        )


def _v5_runtime_class_functions(class_object: type) -> tuple[FunctionType, ...]:
    functions = []
    for owner in class_object.__mro__:
        if owner is object or owner.__module__ != class_object.__module__:
            continue
        for descriptor in vars(owner).values():
            parsed = _v5_class_callable_descriptor(descriptor)
            if parsed is not None:
                functions.extend(function for _, function in parsed[1])
    return tuple(functions)


def _discover_v5_runtime_reachable_global_pins(
    modules: tuple[ModuleType, ...],
) -> tuple[_V5RuntimeReachableGlobalPin, ...]:
    """Resolve the recursive Neural Forge global graph from runtime roots."""

    pending: list[FunctionType] = []
    for module in modules:
        for value in vars(module).values():
            if isinstance(value, FunctionType) and value.__module__ == module.__name__:
                pending.append(value)
            elif isinstance(value, type) and value.__module__ == module.__name__:
                pending.extend(_v5_runtime_class_functions(value))
    visited_functions: set[int] = set()
    visited_classes: set[int] = set()
    bindings: dict[
        tuple[int, str],
        tuple[str, dict[str, Any], str, Any],
    ] = {}
    while pending:
        function = pending.pop()
        if id(function) in visited_functions:
            continue
        visited_functions.add(id(function))
        globals_value = function.__globals__
        owner_module = globals_value.get("__name__")
        if not isinstance(owner_module, str):
            raise RuntimeError("V5 reachable runtime callable has no defining module.")
        for name in function.__code__.co_names:
            if name not in globals_value:
                continue
            binding = globals_value[name]
            key = (id(globals_value), name)
            prior = bindings.get(key)
            if prior is not None and prior[3] is not binding:
                raise RuntimeError("V5 reachable runtime global changed during import.")
            bindings[key] = (
                owner_module,
                globals_value,
                name,
                binding,
            )
            if isinstance(binding, FunctionType) and binding.__module__.startswith(
                "neural_forge."
            ):
                pending.append(binding)
            elif (
                isinstance(binding, type)
                and binding.__module__.startswith("neural_forge.")
                and id(binding) not in visited_classes
            ):
                visited_classes.add(id(binding))
                pending.extend(_v5_runtime_class_functions(binding))
    return tuple(
        _V5RuntimeReachableGlobalPin.create(
            owner_module=owner_module,
            owner_globals=owner_globals,
            name=name,
            binding_object=binding,
        )
        for owner_module, owner_globals, name, binding in sorted(
            bindings.values(),
            key=lambda row: (row[0], row[2]),
        )
    )


_V5_RUNTIME_CALLABLE_PINS = tuple(
    _V5RuntimeCallablePin.create(label, callable_object)
    for label, callable_object in (
        (
            "candidate_replay.cuda_worker",
            _AUTHENTIC_CUDA_REPLAY_WORKER,
        ),
        (
            "candidate_replay.cpu_parity_worker",
            _AUTHENTIC_CPU_PARITY_WORKER,
        ),
        (
            "candidate_replay.payload",
            _AUTHENTIC_CANDIDATE_REPLAY_PAYLOAD,
        ),
        (
            "candidate_replay.evaluator.__init__",
            _AUTHENTIC_TWO_GPU_EVALUATOR_CLASS.__init__,
        ),
        (
            "candidate_replay.evaluator.evaluate",
            _AUTHENTIC_TWO_GPU_EVALUATOR_CLASS.evaluate,
        ),
        (
            "replay_scheduler.__init__",
            _AUTHENTIC_REPLAY_SCHEDULER_CLASS.__init__,
        ),
        (
            "replay_scheduler.authenticate_runtime",
            _AUTHENTIC_REPLAY_SCHEDULER_CLASS.authenticate_runtime,
        ),
        (
            "replay_scheduler.execute",
            _AUTHENTIC_REPLAY_SCHEDULER_CLASS.execute,
        ),
        (
            "replay_scheduler._reauthenticate",
            _AUTHENTIC_REPLAY_SCHEDULER_CLASS._reauthenticate,
        ),
        (
            "replay_scheduler.policy.__post_init__",
            _AUTHENTIC_REPLAY_SCHEDULER_POLICY_CLASS.__post_init__,
        ),
        (
            "replay_scheduler.policy.as_dict",
            _AUTHENTIC_REPLAY_SCHEDULER_POLICY_CLASS.as_dict,
        ),
        (
            "replay_scheduler.capture_device_receipt",
            _AUTHENTIC_DEVICE_RECEIPT_PROBE,
        ),
        (
            "replay_scheduler.state_probes.__init__",
            _AUTHENTIC_REPLAY_STATE_PROBES_CLASS.__init__,
        ),
        (
            "replay_scheduler.state_probes.installed",
            _AUTHENTIC_REPLAY_STATE_PROBES_CLASS.installed_state_probe,
        ),
        (
            "replay_scheduler.state_probes.cpu",
            _AUTHENTIC_REPLAY_STATE_PROBES_CLASS.cpu_installed_state_probe,
        ),
        (
            "replay_scheduler.state_probes.as_dict",
            _AUTHENTIC_REPLAY_STATE_PROBES_CLASS.as_dict,
        ),
        (
            "replay_scheduler.host_factory.__post_init__",
            _AUTHENTIC_PINNED_HOST_FACTORY_CLASS.__post_init__,
        ),
        (
            "replay_scheduler.host_factory.__call__",
            _AUTHENTIC_PINNED_HOST_FACTORY_CLASS.__call__,
        ),
        (
            "replay_scheduler.host_buffer.__init__",
            _AUTHENTIC_PINNED_HOST_BUFFER_CLASS.__init__,
        ),
        (
            "replay_scheduler.host_buffer.stage",
            _AUTHENTIC_PINNED_HOST_BUFFER_CLASS.stage,
        ),
        (
            "replay_scheduler.host_buffer.as_dict",
            _AUTHENTIC_PINNED_HOST_BUFFER_CLASS.as_dict,
        ),
    )
)

_V5_RUNTIME_CLASS_PINS = tuple(
    _V5RuntimeClassPin.create(label, class_object)
    for label, class_object in (
        (
            "candidate_replay.evaluator",
            _AUTHENTIC_TWO_GPU_EVALUATOR_CLASS,
        ),
        (
            "replay_scheduler.scheduler",
            _AUTHENTIC_REPLAY_SCHEDULER_CLASS,
        ),
        (
            "replay_scheduler.policy",
            _AUTHENTIC_REPLAY_SCHEDULER_POLICY_CLASS,
        ),
        (
            "replay_scheduler.state_probes",
            _AUTHENTIC_REPLAY_STATE_PROBES_CLASS,
        ),
        (
            "replay_scheduler.host_factory",
            _AUTHENTIC_PINNED_HOST_FACTORY_CLASS,
        ),
        (
            "replay_scheduler.host_buffer",
            _AUTHENTIC_PINNED_HOST_BUFFER_CLASS,
        ),
    )
)

_V5_RUNTIME_MODULE_PINS = (
    _V5RuntimeModulePin.create(
        "dynamic_replay_scheduler",
        _dynamic_replay_scheduler_module,
    ),
    _V5RuntimeModulePin.create(
        "hf_lfm2_cpcf_candidate_replay",
        _candidate_replay_module,
    ),
    _V5RuntimeModulePin.create(
        "hf_lfm2_cpcf_resume_artifact",
        _resume_artifact_module,
    ),
    _V5RuntimeModulePin.create(
        "hf_lfm2_cpcf_v5_admission_proof",
        _v5_admission_proof_module,
    ),
    _V5RuntimeModulePin.create(
        "hf_lfm2_cpcf_v5_model_compiler",
        _v5_model_compiler_module,
    ),
    _V5RuntimeModulePin.create(
        "hf_lfm2_cpcf_v5_resume",
        _v5_resume_module,
    ),
)
_V5_RUNTIME_REACHABLE_GLOBAL_PINS = _discover_v5_runtime_reachable_global_pins(
    (
        _dynamic_replay_scheduler_module,
        _candidate_replay_module,
        _resume_artifact_module,
        _v5_admission_proof_module,
        _v5_model_compiler_module,
        _v5_resume_module,
    )
)
_V5_RUNTIME_DISCOVERED_HELPER_BINDINGS = tuple(
    (pin.module_object, name)
    for pin in _V5_RUNTIME_MODULE_PINS
    for name in pin.helper_binding_names
)
_V5_RUNTIME_DISCOVERED_REACHABLE_HELPER_BINDINGS = tuple(
    (sys.modules[pin.owner_module], pin.name)
    for pin in _V5_RUNTIME_REACHABLE_GLOBAL_PINS
    if pin.owner_module.startswith("neural_forge.")
    and (
        isinstance(pin.binding_object, FunctionType)
        or (
            isinstance(pin.binding_object, type)
            and pin.binding_object.__module__.startswith("neural_forge.")
        )
    )
)


def _authenticate_v5_runtime_objects(
    *,
    _pins: tuple[_V5RuntimeCallablePin, ...] = _V5_RUNTIME_CALLABLE_PINS,
    _class_pins: tuple[_V5RuntimeClassPin, ...] = _V5_RUNTIME_CLASS_PINS,
    _module_pins: tuple[_V5RuntimeModulePin, ...] = _V5_RUNTIME_MODULE_PINS,
    _reachable_pins: tuple[
        _V5RuntimeReachableGlobalPin,
        ...,
    ] = _V5_RUNTIME_REACHABLE_GLOBAL_PINS,
    _evaluator_class: type = _AUTHENTIC_TWO_GPU_EVALUATOR_CLASS,
    _scheduler_class: type = _AUTHENTIC_REPLAY_SCHEDULER_CLASS,
    _scheduler_policy_class: type = (_AUTHENTIC_REPLAY_SCHEDULER_POLICY_CLASS),
    _state_probes_class: type = _AUTHENTIC_REPLAY_STATE_PROBES_CLASS,
    _host_factory_class: type = _AUTHENTIC_PINNED_HOST_FACTORY_CLASS,
    _host_buffer_class: type = _AUTHENTIC_PINNED_HOST_BUFFER_CLASS,
    _cuda_worker: Callable[..., Any] = _AUTHENTIC_CUDA_REPLAY_WORKER,
    _cpu_worker: Callable[..., Any] = _AUTHENTIC_CPU_PARITY_WORKER,
    _payload_worker: Callable[..., Any] = _AUTHENTIC_CANDIDATE_REPLAY_PAYLOAD,
    _device_probe: Callable[..., Any] = _AUTHENTIC_DEVICE_RECEIPT_PROBE,
) -> Mapping[str, Any]:
    """Reauthenticate exact production CUDA runtime objects and source bytes."""

    for pin in _reachable_pins:
        if pin.owner_globals.get(pin.name) is not pin.binding_object:
            raise Lfm2CpcfCatalogError(
                "V5 reachable runtime global "
                f"{pin.owner_module}.{pin.name} changed after import."
            )
    authenticated_modules = [dict(pin.authenticate()) for pin in _module_pins]
    if (
        AuthenticatedTwoGpuLfm2CpcfCandidateReplayEvaluator is not _evaluator_class
        or DeterministicReplayScheduler is not _scheduler_class
        or ReplaySchedulerPolicy is not _scheduler_policy_class
        or AuthenticatedReplayStateProbes is not _state_probes_class
        or AuthenticatedPinnedHostBufferFactory is not _host_factory_class
        or PinnedHostDoubleBuffer is not _host_buffer_class
        or lfm2_cpcf_cuda_replay_worker is not _cuda_worker
        or lfm2_cpcf_cpu_parity_worker is not _cpu_worker
        or replay_lfm2_cpcf_candidate_work_payload is not _payload_worker
        or capture_cuda_replay_device_receipt is not _device_probe
        or _candidate_replay_module.AuthenticatedTwoGpuLfm2CpcfCandidateReplayEvaluator
        is not _evaluator_class
        or _candidate_replay_module.lfm2_cpcf_cuda_replay_worker is not _cuda_worker
        or _candidate_replay_module.lfm2_cpcf_cpu_parity_worker is not _cpu_worker
        or _candidate_replay_module.replay_lfm2_cpcf_candidate_work_payload
        is not _payload_worker
        or _dynamic_replay_scheduler_module.DeterministicReplayScheduler
        is not _scheduler_class
        or _dynamic_replay_scheduler_module.ReplaySchedulerPolicy
        is not _scheduler_policy_class
        or _dynamic_replay_scheduler_module.AuthenticatedReplayStateProbes
        is not _state_probes_class
        or _dynamic_replay_scheduler_module.AuthenticatedPinnedHostBufferFactory
        is not _host_factory_class
        or _dynamic_replay_scheduler_module.PinnedHostDoubleBuffer
        is not _host_buffer_class
        or _dynamic_replay_scheduler_module.capture_cuda_replay_device_receipt
        is not _device_probe
    ):
        raise Lfm2CpcfCatalogError(
            "The production V5 evaluator, scheduler, worker, probe, or "
            "pinned-host factory object changed after import."
        )
    expected_methods = {
        "candidate_replay.evaluator.__init__": _evaluator_class.__init__,
        "candidate_replay.evaluator.evaluate": _evaluator_class.evaluate,
        "replay_scheduler.__init__": _scheduler_class.__init__,
        "replay_scheduler.authenticate_runtime": (
            _scheduler_class.authenticate_runtime
        ),
        "replay_scheduler.execute": _scheduler_class.execute,
        "replay_scheduler._reauthenticate": _scheduler_class._reauthenticate,
        "replay_scheduler.policy.__post_init__": (
            _scheduler_policy_class.__post_init__
        ),
        "replay_scheduler.policy.as_dict": (_scheduler_policy_class.as_dict),
        "replay_scheduler.state_probes.__init__": (_state_probes_class.__init__),
        "replay_scheduler.state_probes.installed": (
            _state_probes_class.installed_state_probe
        ),
        "replay_scheduler.state_probes.cpu": (
            _state_probes_class.cpu_installed_state_probe
        ),
        "replay_scheduler.state_probes.as_dict": (_state_probes_class.as_dict),
        "replay_scheduler.host_factory.__call__": (_host_factory_class.__call__),
        "replay_scheduler.host_factory.__post_init__": (
            _host_factory_class.__post_init__
        ),
        "replay_scheduler.host_buffer.__init__": (_host_buffer_class.__init__),
        "replay_scheduler.host_buffer.stage": _host_buffer_class.stage,
        "replay_scheduler.host_buffer.as_dict": _host_buffer_class.as_dict,
    }
    for pin in _pins:
        if pin.label in expected_methods and (
            pin.callable_object is not expected_methods[pin.label]
        ):
            raise Lfm2CpcfCatalogError(
                f"Production V5 method {pin.label!r} changed after import."
            )
    authenticated = [dict(pin.authenticate()) for pin in _pins]
    authenticated_classes = [dict(pin.authenticate()) for pin in _class_pins]
    authenticated_reachable = [dict(pin.authenticate()) for pin in _reachable_pins]
    body = {
        "schema": "neural-forge-lfm2-cpcf-v5-runtime-object-pins-v4",
        "callables": authenticated,
        "classes": authenticated_classes,
        "modules": authenticated_modules,
        "reachable_globals": authenticated_reachable,
    }
    return MappingProxyType({**body, "report_sha256": canonical_sha256(body)})


def _authenticate_v5_compiler_callable() -> Mapping[str, Any]:
    """Authenticate the exact compiler and its reachable resume implementation."""

    current = compile_lfm2_cpcf_v5_selective_model_fold
    if (
        current is not _AUTHENTIC_V5_COMPILER_CALLABLE
        or not _AUTHENTIC_V5_COMPILER_SOURCE_PATH.is_file()
        or hashlib.sha256(marshal.dumps(current.__code__)).hexdigest()
        != _AUTHENTIC_V5_COMPILER_CODE_SHA256
        or _file_sha256(_AUTHENTIC_V5_COMPILER_SOURCE_PATH)
        != _AUTHENTIC_V5_COMPILER_SOURCE_SHA256
    ):
        raise Lfm2CpcfCatalogError(
            "The production v5 compiler callable changed after import."
        )
    compiler_modules = frozenset(
        {
            _resume_artifact_module.__name__,
            _v5_admission_proof_module.__name__,
            _v5_model_compiler_module.__name__,
            _v5_resume_module.__name__,
        }
    )
    authenticated_modules = [
        dict(pin.authenticate())
        for pin in _V5_RUNTIME_MODULE_PINS
        if pin.module_object.__name__ in compiler_modules
    ]
    authenticated_reachable = [
        dict(pin.authenticate())
        for pin in _V5_RUNTIME_REACHABLE_GLOBAL_PINS
        if pin.owner_module in compiler_modules
    ]
    body = {
        "module": current.__module__,
        "qualname": current.__qualname__,
        "code_sha256": _AUTHENTIC_V5_COMPILER_CODE_SHA256,
        "source_path": str(_AUTHENTIC_V5_COMPILER_SOURCE_PATH),
        "source_sha256": _AUTHENTIC_V5_COMPILER_SOURCE_SHA256,
        "implementation_modules": authenticated_modules,
        "reachable_globals": authenticated_reachable,
    }
    return MappingProxyType({**body, "authentication_sha256": canonical_sha256(body)})


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2CpcfCatalogError(
            "CPCF catalog metadata must be finite canonical JSON."
        ) from exc


def _tensor_sha256(value: Any, name: str) -> str:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
    ):
        raise Lfm2CpcfCatalogError(f"{name} must be a materialized dense tensor.")
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(size) for size in tensor.shape],
    }
    digest = hashlib.sha256(_canonical_json(descriptor))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(child) for child in value]
    return value


def _v5_resume_selection(
    coordinator: Lfm2CpcfV5ResumeCoordinator | None,
    state: Lfm2CpcfV5ResumeState | None,
    admission_sha256: str | None,
) -> tuple[
    Lfm2CpcfV5ResumeCoordinator | None,
    Lfm2CpcfV5ResumeState | None,
    str | None,
    Mapping[str, Any],
]:
    """Normalize one cold publication store or one admitted continuation."""

    if coordinator is None:
        if state is not None or admission_sha256 is not None:
            raise Lfm2CpcfCatalogError(
                "A V5 resume state/reference requires an exact typed "
                "resume coordinator."
            )
        body = {
            "schema": LFM2_CPCF_V5_RESUME_SCHEMA,
            "resume_publication_enabled": False,
            "resume_used": False,
            "coordinator_store_binding_sha256": None,
            "admission_sha256": None,
            "next_layer_index": None,
            "frontier_sha256": None,
            "context_sha256": None,
            "frontier_reference": None,
            "checkpoint_reference": None,
        }
        return None, None, None, MappingProxyType(body)
    if type(coordinator) is not Lfm2CpcfV5ResumeCoordinator:
        raise Lfm2CpcfCatalogError(
            "resume_coordinator must have the exact V5 coordinator type."
        )
    root = Path(coordinator.store.root).resolve()
    store_binding = canonical_sha256(
        {
            "schema": LFM2_CPCF_V5_RESUME_SCHEMA,
            "store_type": (
                f"{type(coordinator.store).__module__}."
                f"{type(coordinator.store).__qualname__}"
            ),
            "resolved_root": str(root),
        }
    )
    if state is None and admission_sha256 is None:
        body = {
            "schema": LFM2_CPCF_V5_RESUME_SCHEMA,
            "resume_publication_enabled": True,
            "resume_used": False,
            "coordinator_store_binding_sha256": store_binding,
            "admission_sha256": None,
            "next_layer_index": None,
            "frontier_sha256": None,
            "context_sha256": None,
            "frontier_reference": None,
            "checkpoint_reference": None,
        }
        return coordinator, None, None, MappingProxyType(body)
    if state is not None and type(state) is not Lfm2CpcfV5ResumeState:
        raise Lfm2CpcfCatalogError(
            "V5 continuation requires an admitted typed V5 resume state; "
            "pending, rejected, untyped, and V4 checkpoints fail closed."
        )
    supplied_admission = (
        admission_sha256
        if admission_sha256 is not None
        else state.admission.get("admission_sha256")
    )
    admitted_sha = _digest(
        supplied_admission,
        "resume_admission_sha256",
    )
    try:
        reopened = coordinator.open_admitted(admitted_sha)
    except Exception as exc:
        raise Lfm2CpcfCatalogError(
            "V5 continuation reference is not an admitted authenticated "
            "prefix; pending and rejected checkpoints are not resumable."
        ) from exc
    if state is not None and (
        state.coordinator is not coordinator
        or state.frontier_reference != reopened.frontier_reference
        or state.checkpoint_reference != reopened.checkpoint_reference
        or state.frontier.frontier_sha256 != reopened.frontier.frontier_sha256
        or state.next_layer_index != reopened.next_layer_index
        or state.admission.get("admission_sha256") != admitted_sha
    ):
        raise Lfm2CpcfCatalogError(
            "Supplied V5 resume state, coordinator, and admission reference "
            "are not the same authenticated prefix."
        )
    body = {
        "schema": LFM2_CPCF_V5_RESUME_SCHEMA,
        "resume_publication_enabled": True,
        "resume_used": True,
        "coordinator_store_binding_sha256": store_binding,
        "admission_sha256": admitted_sha,
        "next_layer_index": reopened.next_layer_index,
        "frontier_sha256": reopened.frontier.frontier_sha256,
        "context_sha256": reopened.frontier.context["context_sha256"],
        "frontier_reference": reopened.frontier_reference.as_dict(),
        "checkpoint_reference": reopened.checkpoint_reference.as_dict(),
    }
    return (
        coordinator,
        reopened,
        admitted_sha,
        MappingProxyType(body),
    )


def _verify_v5_resume_static_context(
    state: Lfm2CpcfV5ResumeState | None,
    *,
    backend: Lfm2CpcfV5SelectiveBackend,
    trace: Lfm2StrictTokenizerBridgeTraceAdapter,
    source_tied_table: Any,
    reviewed: Lfm2CpcfReviewedPromotionEvidence,
) -> None:
    if state is None:
        return
    context = state.frontier.context
    try:
        selector_sha = _optional_runtime_contract_sha256(
            backend.candidate_selector,
            kind="candidate_selector",
        )
        gpu_sha = _optional_runtime_contract_sha256(
            backend.candidate_replay_evaluator,
            kind="candidate_replay",
        )
        numerical_sha = _optional_runtime_contract_sha256(
            backend.numerical_frontier_adapter,
            kind="numerical_frontier",
        )
    except Exception as exc:
        raise Lfm2CpcfCatalogError(
            "V5 resume runtime contracts could not be authenticated."
        ) from exc
    expected = {
        "source_model_state_sha256": (reviewed.live_parent_state_fingerprint),
        "cache_sha256": backend.cache.cache_key,
        "selective_catalog_sha256": (backend.expected_selective_catalog_sha256),
        "selective_source_sha256": (backend.expected_selective_source_sha256),
        "selective_provider_sha256": (backend.expected_selective_provider_sha256),
        "trace_sha256": trace.evidence_sha256,
        "tokenizer_sha256": (trace.target_tokenizer_package_fingerprint),
        "token_bridge_sha256": trace.bridge_artifact_sha256,
        "source_tied_table_sha256": _tensor_sha256(
            source_tied_table,
            "source_tied_table",
        ),
        "blank_target_state_sha256": (reviewed.blank_target_state_fingerprint),
        "planned_target_unique_parameters": (reviewed.exact_target_unique_parameters),
        "runtime_policy_sha256": canonical_sha256(backend.runtime_policy.as_dict()),
        "physical_rss_policy_sha256": (
            build_lfm2_cpcf_v5_physical_rss_policy_sha256(
                full_phase_tree_logical_bytes=(
                    backend.runtime_policy.full_phase_tree_logical_bytes
                ),
                minimum_reduction_fraction=(
                    backend.runtime_policy.minimum_physical_rss_reduction_fraction
                ),
                maximum_increase_bytes=(
                    backend.runtime_policy.maximum_physical_rss_increase_bytes
                ),
                maximum_process_rss_bytes=(
                    backend.runtime_policy.maximum_process_rss_bytes
                ),
            )
        ),
        "compiler_implementation_sha256": (
            lfm2_cpcf_v5_compiler_implementation()["implementation_sha256"]
        ),
        "resume_implementation_sha256": (
            lfm2_cpcf_v5_resume_implementation()["implementation_sha256"]
        ),
        "selector_contract_sha256": selector_sha,
        "numerical_contract_sha256": numerical_sha,
        "gpu_contract_sha256": gpu_sha,
    }
    if state.frontier.expected_layer_count != backend.expected_layer_count or any(
        context.get(name) != value for name, value in expected.items()
    ):
        raise Lfm2CpcfCatalogError(
            "Admitted V5 resume context differs from its exact parent, "
            "cache, trace/bridge, blank, runtime policy, compiler, "
            "numerical, selector, or GPU production resources."
        )


def _phase_body(phase: Lfm2CpcfTargetPhase) -> dict[str, Any]:
    return {
        "schema": "neural-forge-lfm2-cpcf-target-phase-v1",
        "batch_index": phase.batch_index,
        "record_id": phase.record_id,
        "split": phase.split,
        "phase": phase.phase,
        "donor_input_sha256": phase.donor_input_sha256,
        "target_alignment_sha256": phase.target_alignment_sha256,
        "input_ids_sha256": _tensor_sha256(
            phase.input_ids,
            "phase.input_ids",
        ),
        "attention_mask_sha256": _tensor_sha256(
            phase.attention_mask,
            "phase.attention_mask",
        ),
        "position_ids_sha256": _tensor_sha256(
            phase.position_ids,
            "phase.position_ids",
        ),
        "batch_size": phase.batch_size,
        "full_prefix_tokens": phase.full_prefix_tokens,
        "sampled_suffix_tokens": phase.sampled_suffix_tokens,
        "parent_evidence_sha256": phase.parent_evidence.evidence_sha256,
        "numeric_token_id_equality_required": False,
        "actual_child_full_prefix_required": True,
    }


def _phase_metadata(phase: Lfm2CpcfTargetPhase) -> dict[str, Any]:
    body = _phase_body(phase)
    observed_sha = canonical_sha256(body)
    if observed_sha != phase.phase_sha256:
        raise Lfm2CpcfCatalogError(
            f"CPCF target phase {phase.identity!r} failed its metadata seal."
        )
    return {
        **phase.as_dict(),
        "phase_sha256": observed_sha,
    }


def _parent_tensor_path(
    phase: Lfm2CpcfTargetPhase,
    *,
    layer_index: int | None,
    field_name: str,
) -> str:
    prefix = (
        f"phase[{phase.batch_index}:{phase.phase}]"
        if layer_index is None
        else (f"phase[{phase.batch_index}:{phase.phase}].layer[{layer_index}]")
    )
    return f"{prefix}.{field_name}"


def _parent_tensor_bindings(
    phases: Sequence[Lfm2CpcfTargetPhase],
) -> tuple[tuple[str, Any], ...]:
    bindings: list[tuple[str, Any]] = []
    for phase in phases:
        evidence = phase.parent_evidence
        for layer in evidence.layers:
            bindings.extend(
                (
                    _parent_tensor_path(
                        phase,
                        layer_index=layer.layer_index,
                        field_name=str(name),
                    ),
                    tensor,
                )
                for name, tensor in sorted(layer.tensors.items())
            )
        bindings.extend(
            (
                _parent_tensor_path(
                    phase,
                    layer_index=None,
                    field_name=str(name),
                ),
                tensor,
            )
            for name, tensor in sorted(evidence.final_norm_tensors.items())
        )
    paths = tuple(path for path, _tensor in bindings)
    if len(set(paths)) != len(paths):
        raise Lfm2CpcfCatalogError(
            "CPCF parent evidence exposes duplicate tensor paths."
        )
    return tuple(bindings)


def _authenticate_parent_phase_tensors(
    phase: Lfm2CpcfTargetPhase,
) -> dict[str, str]:
    """Recompute the canonical installability receipts for one phase.

    ``as_dict`` is intentionally used here because the installability
    evidence classes own the canonical tensor-SHA implementation.  This keeps
    the catalog seal byte-for-byte aligned with the cache authenticator.
    """

    evidence = phase.parent_evidence
    document = evidence.as_dict()
    layer_documents = document.get("layers")
    if (
        document.get("cache_key") != evidence.cache_key
        or document.get("batch_index") != evidence.batch_index
        or document.get("input_sha256") != evidence.input_sha256
        or document.get("phase") != evidence.phase
        or not isinstance(layer_documents, list)
        or len(layer_documents) != len(evidence.layers)
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF parent phase document differs from its typed evidence."
        )
    authenticated: dict[str, str] = {}
    layer_sha: list[str] = []
    for layer, layer_document in zip(
        evidence.layers,
        layer_documents,
        strict=True,
    ):
        inventory = layer_document.get("field_inventory")
        if (
            layer_document.get("layer_index") != layer.layer_index
            or layer_document.get("operator_kind") != layer.operator_kind
            or layer_document.get("source_width") != layer.source_width
            or not isinstance(inventory, Mapping)
            or set(inventory) != set(layer.tensors)
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF parent layer inventory differs from its tensor bank."
            )
        body = {
            "schema": layer_document.get("schema"),
            "cache_key": evidence.cache_key,
            "batch_index": evidence.batch_index,
            "phase": evidence.phase,
            "layer_index": layer.layer_index,
            "operator_kind": layer.operator_kind,
            "source_width": layer.source_width,
            "field_inventory": dict(inventory),
            "functional_identities_verified": True,
        }
        observed_layer_sha = canonical_sha256(body)
        if observed_layer_sha != layer.evidence_sha256:
            raise Lfm2CpcfCatalogError(
                "CPCF parent layer tensor authentication changed."
            )
        layer_sha.append(observed_layer_sha)
        for name, row in inventory.items():
            if not isinstance(row, Mapping):
                raise Lfm2CpcfCatalogError(
                    "CPCF parent tensor inventory row is invalid."
                )
            authenticated[
                _parent_tensor_path(
                    phase,
                    layer_index=layer.layer_index,
                    field_name=str(name),
                )
            ] = _digest(
                row.get("sha256"),
                "parent tensor inventory sha256",
            )
    final_inventory = document.get("final_norm_field_inventory")
    if not isinstance(final_inventory, Mapping) or set(final_inventory) != set(
        evidence.final_norm_tensors
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF final-normalization inventory differs from its tensor bank."
        )
    for name, row in final_inventory.items():
        if not isinstance(row, Mapping):
            raise Lfm2CpcfCatalogError(
                "CPCF final-normalization inventory row is invalid."
            )
        authenticated[
            _parent_tensor_path(
                phase,
                layer_index=None,
                field_name=str(name),
            )
        ] = _digest(
            row.get("sha256"),
            "final-normalization tensor inventory sha256",
        )
    phase_body = {
        "schema": document.get("schema"),
        "cache_key": evidence.cache_key,
        "batch_index": evidence.batch_index,
        "input_sha256": evidence.input_sha256,
        "phase": evidence.phase,
        "layer_evidence_sha256": layer_sha,
        "final_norm_field_inventory": dict(final_inventory),
        "parent_side_compiler_inputs_ready": True,
    }
    if canonical_sha256(phase_body) != evidence.evidence_sha256:
        raise Lfm2CpcfCatalogError("CPCF parent phase tensor authentication changed.")
    return authenticated


def _authenticate_parent_tensors(
    phases: Sequence[Lfm2CpcfTargetPhase],
) -> Mapping[str, str]:
    authenticated: dict[str, str] = {}
    for phase in phases:
        phase_tensors = _authenticate_parent_phase_tensors(phase)
        overlap = set(authenticated).intersection(phase_tensors)
        if overlap:
            raise Lfm2CpcfCatalogError(
                "CPCF parent tensor authentication paths overlap."
            )
        authenticated.update(phase_tensors)
    if set(authenticated) != {
        path for path, _tensor in _parent_tensor_bindings(phases)
    }:
        raise Lfm2CpcfCatalogError("CPCF parent tensor authentication is incomplete.")
    return MappingProxyType(authenticated)


@dataclass(frozen=True, slots=True)
class _ParentTensorMutationSeal:
    path: str
    object_id: int
    storage_object_id: int
    storage_data_ptr: int
    storage_nbytes: int
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str
    version: int
    content_sha256: str


def _parent_tensor_mutation_seal(
    path: str,
    tensor: Any,
    *,
    content_sha256: str,
) -> _ParentTensorMutationSeal:
    torch = _torch()
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.layout != torch.strided
        or tensor.device.type == "meta"
    ):
        raise Lfm2CpcfCatalogError(
            f"CPCF parent evidence tensor {path!r} is not materialized dense storage."
        )
    storage = tensor.untyped_storage()
    version = getattr(tensor, "_version", None)
    if isinstance(version, bool) or not isinstance(version, int):
        raise Lfm2CpcfCatalogError(
            f"CPCF parent evidence tensor {path!r} exposes no mutation version counter."
        )
    return _ParentTensorMutationSeal(
        path=path,
        object_id=id(tensor),
        storage_object_id=id(storage),
        storage_data_ptr=int(storage.data_ptr()),
        storage_nbytes=int(storage.nbytes()),
        storage_offset=int(tensor.storage_offset()),
        shape=tuple(int(size) for size in tensor.shape),
        stride=tuple(int(value) for value in tensor.stride()),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        version=version,
        content_sha256=_digest(
            content_sha256,
            f"parent tensor {path} content_sha256",
        ),
    )


def _blank_report(
    target: DynamicTargetSynthesisResult,
    selected: Lfm2DenseBudgetedSelectedTarget,
) -> Mapping[str, Any]:
    try:
        topology = verify_compiler_authenticated_target(target)
    except DynamicTargetSynthesisError as exc:
        raise Lfm2CpcfCatalogError(
            "CPCF target lacks compiler-authenticated blank provenance."
        ) from exc
    provenance = target.blank_provenance
    if (
        provenance is None
        or not provenance.compiler_created_blank
        or not provenance.all_parameters_zero
        or not provenance.dense_materialized_parameters
        or not target.topology_provenance_verified
        or fingerprint_model_state(target.model) != target.blank_state_fingerprint
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF requires the live compiler-created dense zero target."
        )
    body = {
        "schema": LFM2_CPCF_BLANK_REPORT_SCHEMA,
        "compiler_created_blank": True,
        "all_parameters_zero": True,
        "dense_materialized_parameters": True,
        "topology_provenance_verified": True,
        "weights_installed": False,
        "training_used": False,
        "quantization_used": False,
        "pruning_used": False,
        "target_unique_parameters": (selected.exact_target_unique_parameters),
        "target_graph_fingerprint": target.graph.fingerprint,
        "target_topology_fingerprint": topology,
        "target_blank_state_fingerprint": target.blank_state_fingerprint,
        "target_blank_provenance_digest": target.blank_provenance_digest,
        "source_graph_fingerprint": selected.source_graph_fingerprint,
        "source_state_fingerprint": selected.source_state_fingerprint,
    }
    return MappingProxyType({**body, "report_sha256": canonical_sha256(body)})


@dataclass(frozen=True, slots=True)
class Lfm2CpcfExternalBehaviorGateContract:
    """Proof that promotion is independent, free-running, and semantic.

    This is deliberately a *reviewed promotion-rerun* contract.  A candidate's
    manual packet digest cannot exist before the first fold.  That first live
    run must therefore remain quarantined and use the packet-capture workflow;
    it cannot instantiate this production stack or publish.  Only a
    deterministic rerun bound to the authenticated reviewed packet can pass
    stack construction and reach the publication gates.
    """

    independent_holdout_manifest_sha256: str
    independent_parent_envelope_sha256: str
    parent_relative_manual_packet_sha256: str
    independent_holdout_case_count: int = 64
    candidate_free_generation: bool = True
    parent_relative_manual_scoring: bool = True
    forced_decode_only: bool = False
    exact_text_match_required: bool = False
    contract_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "independent_holdout_manifest_sha256",
            "independent_parent_envelope_sha256",
            "parent_relative_manual_packet_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), name),
            )
        if (
            isinstance(self.independent_holdout_case_count, bool)
            or not isinstance(self.independent_holdout_case_count, int)
            or self.independent_holdout_case_count < 64
            or self.candidate_free_generation is not True
            or self.parent_relative_manual_scoring is not True
            or self.forced_decode_only is not False
            or self.exact_text_match_required is not False
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF promotion requires at least 64 independent free-"
                "generation cases plus parent-relative semantic manual review."
            )
        object.__setattr__(
            self,
            "contract_sha256",
            canonical_sha256(self.body()),
        )

    def body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CPCF_EXTERNAL_GATE_SCHEMA,
            "lifecycle": "reviewed-promotion-rerun",
            "first_fold_quarantine_required": True,
            "first_fold_publication_allowed": False,
            "independent_holdout_manifest_sha256": (
                self.independent_holdout_manifest_sha256
            ),
            "independent_parent_envelope_sha256": (
                self.independent_parent_envelope_sha256
            ),
            "parent_relative_manual_packet_sha256": (
                self.parent_relative_manual_packet_sha256
            ),
            "independent_holdout_case_count": (self.independent_holdout_case_count),
            "candidate_free_generation": self.candidate_free_generation,
            "parent_relative_manual_scoring": (self.parent_relative_manual_scoring),
            "forced_decode_only": self.forced_decode_only,
            "exact_text_match_required": self.exact_text_match_required,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.body(), "contract_sha256": self.contract_sha256}


def _strict_behavior_binding(
    value: Any,
    contract: Lfm2CpcfExternalBehaviorGateContract,
    *,
    name: str,
) -> BehaviorGateBinding:
    if type(value) is not Lfm2CpcfReviewedPromotionGate:
        raise Lfm2CpcfCatalogError(
            f"{name} must be the exact typed CPCF reviewed-promotion gate."
        )
    try:
        authenticate_lfm2_cpcf_reviewed_promotion_gate(value)
    except Lfm2CpcfReviewedPromotionError as exc:
        raise Lfm2CpcfCatalogError(
            f"{name} is not a constructor-authenticated immutable reviewed "
            "promotion gate."
        ) from exc
    evidence = getattr(value, "evidence", None)
    if not isinstance(evidence, Lfm2CpcfReviewedPromotionEvidence):
        raise Lfm2CpcfCatalogError(f"{name} has no typed reviewed-promotion evidence.")
    reviewed = dict(evidence.reviewed_evidence)
    suite_conclusion = reviewed.get("holistic_suite_conclusion")
    if (
        reviewed.get("case_count") != contract.independent_holdout_case_count
        or reviewed.get("manifest_sha256")
        != contract.independent_holdout_manifest_sha256
        or reviewed.get("parent_envelope_sha256")
        != contract.independent_parent_envelope_sha256
        or reviewed.get("parent_relative_manual_packet_sha256")
        != contract.parent_relative_manual_packet_sha256
        or reviewed.get("parent_relative_manual_diagnostics_complete") is not True
        or reviewed.get("parent_relative_manual_metrics_are_diagnostic_only")
        is not True
        or reviewed.get("qualitative_manual_review_complete") is not True
        or reviewed.get("qualitative_manual_review_case_count")
        != contract.independent_holdout_case_count
        or not isinstance(reviewed.get("qualitative_case_reviews_sha256"), str)
        or re.fullmatch(
            r"[0-9a-f]{64}",
            reviewed["qualitative_case_reviews_sha256"],
        )
        is None
        or not isinstance(suite_conclusion, Mapping)
        or suite_conclusion.get("conclusion") != PROMOTION_ELIGIBLE_SUITE_CONCLUSION
        or reviewed.get("holistic_suite_conclusion_sha256")
        != canonical_sha256(suite_conclusion)
        or reviewed.get("qualitative_judgment_is_authoritative") is not True
        or reviewed.get("numeric_scores_and_thresholds_are_diagnostic_only") is not True
        or reviewed.get("automatic_gate_ready_for_manual_completion") is not True
    ):
        raise Lfm2CpcfCatalogError(
            f"{name} reviewed evidence differs from the external promotion contract."
        )
    try:
        binding = resolve_behavior_gate_binding(value)
    except DynamicBehaviorGateError as exc:
        raise Lfm2CpcfCatalogError(
            f"{name} must expose a canonical behavior-gate manifest."
        ) from exc
    components = binding.manifest.get("components")
    reviewed_components = (
        []
        if not isinstance(components, list)
        else [
            component
            for component in components
            if isinstance(component, Mapping)
            and component.get("schema") == REVIEWED_EVIDENCE_SCHEMA
        ]
    )
    if (
        binding.baseline_fingerprint != evidence.full_reviewed_evidence_fingerprint
        or len(reviewed_components) != 1
        or reviewed_components[0].get("reviewed_evidence_fingerprint")
        != evidence.reviewed_evidence_fingerprint
        or reviewed_components[0].get("candidate_state_fingerprint")
        != evidence.candidate_state_fingerprint
        or reviewed_components[0].get("candidate_outputs_sha256")
        != evidence.candidate_outputs_sha256
    ):
        raise Lfm2CpcfCatalogError(
            f"{name} behavior manifest is not bound to the exact reviewed "
            "candidate state, outputs, and evidence chain."
        )
    return binding


def _authenticate_cpcf_resource_coverage(
    *,
    coverage_binding: Lfm2CpcfCachePhaseCoverageBinding,
    phases: Sequence[Lfm2CpcfTargetPhase],
    trace_adapter: Lfm2StrictTokenizerBridgeTraceAdapter,
    parent_graph_fingerprint: str,
    policy: Lfm2CpcfModelFoldPolicy,
    reviewed: Lfm2CpcfReviewedPromotionEvidence,
) -> dict[str, Any]:
    """Cross-bind one exact cache/trace/phase root to reviewed compilation."""

    # Local import avoids a module-load cycle: the production resource factory
    # imports this adapter in order to construct Lfm2CpcfStackResources.
    from .dynamic_lfm2_cpcf_resources import (
        Lfm2CpcfCachePhaseCoverageBinding,
    )

    if type(coverage_binding) is not Lfm2CpcfCachePhaseCoverageBinding:
        raise Lfm2CpcfCatalogError(
            "coverage_binding must be the exact typed CPCF cache-phase "
            "coverage binding."
        )
    if canonical_sha256(
        coverage_binding.body()
    ) != coverage_binding.binding_fingerprint or any(
        canonical_sha256(record.body()) != record.record_fingerprint
        for record in coverage_binding.records
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF cache-phase coverage binding failed its canonical seal."
        )

    ordered_phases = tuple(phases)
    phase_sha256 = tuple(
        _phase_metadata(phase)["phase_sha256"] for phase in ordered_phases
    )
    parent_evidence_sha256 = tuple(
        _digest(
            phase.parent_evidence.evidence_sha256,
            "phase parent_evidence.evidence_sha256",
        )
        for phase in ordered_phases
    )
    phase_identities = tuple(phase.identity for phase in ordered_phases)
    phase_cache_keys = {
        _digest(
            phase.parent_evidence.cache_key,
            "phase parent_evidence.cache_key",
        )
        for phase in ordered_phases
    }
    if len(phase_cache_keys) != 1:
        raise Lfm2CpcfCatalogError("CPCF stack resources must bind one parent cache.")
    cache_key = phase_cache_keys.pop()
    coverage_phase_sha256 = tuple(
        value
        for record in coverage_binding.records
        for value in record.target_phase_sha256
    )
    coverage_parent_evidence_sha256 = tuple(
        value
        for record in coverage_binding.records
        for value in record.parent_evidence_sha256
    )
    coverage_phase_identities = tuple(
        (record.ordinal, phase_name)
        for record in coverage_binding.records
        for phase_name in record.phase_names
    )
    trace_records = tuple(trace_adapter.records)
    if (
        coverage_binding.cache_key != cache_key
        or coverage_binding.cache_key != reviewed.parent_cache_key
        or coverage_binding.cache_parent_fingerprint
        != reviewed.cache_live_parent_fingerprint
        or coverage_binding.parent_graph_fingerprint != parent_graph_fingerprint
        or coverage_binding.parent_graph_fingerprint
        != reviewed.blank_target_parent_graph_fingerprint
        or coverage_binding.trace_adapter_sha256 != trace_adapter.evidence_sha256
        or coverage_binding.trace_adapter_sha256 != reviewed.trace_adapter_sha256
        or coverage_binding.trajectory_corpus_sha256
        != reviewed.trajectory_corpus_sha256
        or coverage_binding.cache_batch_count != trace_adapter.record_count
        or tuple(record.record_id for record in coverage_binding.records)
        != tuple(record.record_id for record in trace_records)
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF coverage cache, parent identity, graph, trace adapter, or "
            "trajectory corpus differs from the stack and reviewed evidence."
        )
    if (
        coverage_binding.phase_count != len(phase_sha256)
        or coverage_phase_identities != phase_identities
        or coverage_phase_sha256 != phase_sha256
        or coverage_parent_evidence_sha256 != parent_evidence_sha256
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF coverage does not bind the exact ordered resource phase "
            "identities, target seals, and parent-evidence seals."
        )

    policy_sha256 = canonical_sha256(policy.as_dict())
    reviewed_phase_sha256 = tuple(reviewed.compiled_model_fold_phase_sha256)
    phase_binding_sha256 = canonical_sha256(
        {
            "schema": ORDERED_PHASE_BINDING_SCHEMA,
            "phase_count": len(phase_sha256),
            "ordered_phase_sha256": list(phase_sha256),
        }
    )
    if (
        reviewed.compiled_model_fold_phase_count != len(phase_sha256)
        or reviewed_phase_sha256 != phase_sha256
        or reviewed.compiled_model_fold_phase_binding_sha256 != phase_binding_sha256
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF resource phases differ from the reviewed compiler phase "
            "sequence or binding seal."
        )
    if (
        reviewed.compiled_model_fold_policy_sha256 != policy_sha256
        or canonical_sha256(dict(reviewed.compiled_model_fold_policy)) != policy_sha256
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF resource policy differs from the reviewed compiler policy seal."
        )

    return {
        "coverage_binding_fingerprint": (coverage_binding.binding_fingerprint),
        "coverage_cache_manifest_sha256": (coverage_binding.cache_manifest_sha256),
        "coverage_cache_parent_fingerprint": (
            coverage_binding.cache_parent_fingerprint
        ),
        "coverage_cache_corpus_fingerprint": (
            coverage_binding.cache_corpus_fingerprint
        ),
        "coverage_trajectory_corpus_sha256": (
            coverage_binding.trajectory_corpus_sha256
        ),
        "coverage_phase_count": coverage_binding.phase_count,
        "reviewed_compiled_model_fold_result_sha256": _digest(
            reviewed.compiled_model_fold_result_sha256,
            "reviewed compiled model-fold result SHA-256",
        ),
        "reviewed_compiled_model_fold_policy_sha256": _digest(
            reviewed.compiled_model_fold_policy_sha256,
            "reviewed compiled model-fold policy SHA-256",
        ),
        "reviewed_compiled_model_fold_phase_binding_sha256": _digest(
            reviewed.compiled_model_fold_phase_binding_sha256,
            "reviewed compiled model-fold phase binding SHA-256",
        ),
        "reviewed_compiled_blank_target_report_sha256": _digest(
            reviewed.compiled_blank_target_report_sha256,
            "reviewed compiled blank-target report SHA-256",
        ),
        "reviewed_compiled_parameter_receipt_sha256": _digest(
            reviewed.compiled_parameter_receipt_sha256,
            "reviewed compiled parameter receipt SHA-256",
        ),
    }


def _v5_gpu_parity_source_sha256() -> Mapping[str, str]:
    repository = Path(__file__).resolve().parents[2]
    paths = {
        "dynamic_replay_scheduler.py": (
            repository / "src" / "neural_forge" / "dynamic_replay_scheduler.py"
        ),
        "hf_lfm2_cpcf_candidate_replay.py": (
            repository / "src" / "neural_forge" / "hf_lfm2_cpcf_candidate_replay.py"
        ),
        "verify_lfm2_cpcf_candidate_replay_gpu_parity.py": (
            repository
            / "experiments"
            / "verify_lfm2_cpcf_candidate_replay_gpu_parity.py"
        ),
    }
    if set(paths) != _V5_GPU_PARITY_SOURCE_FILENAMES or any(
        not path.is_file() or path.is_symlink() for path in paths.values()
    ):
        raise Lfm2CpcfCatalogError("V5 GPU parity source files are missing or unsafe.")
    return MappingProxyType(
        {name: _file_sha256(path.resolve()) for name, path in paths.items()}
    )


def _authenticate_v5_gpu_parity_receipt_against_authority(
    value: Any,
    *,
    expected_scheduler_policy: Mapping[str, Any],
    authority: _V5GpuParityAuthority,
    repository_root: Path,
) -> Mapping[str, Any]:
    """Authenticate one file against an explicit immutable trust authority."""

    if type(authority) is not _V5GpuParityAuthority:
        raise Lfm2CpcfCatalogError(
            "V5 GPU parity approval authority has the wrong type."
        )
    try:
        requested_root = repository_root
        trusted_root = repository_root.resolve(strict=True)
        relative_path = Path(authority.repository_relative_path)
        requested_approved = trusted_root / relative_path
        approved = requested_approved.resolve(strict=True)
    except OSError as exc:
        raise Lfm2CpcfCatalogError(
            "Approved V5 GPU parity artifact does not exist."
        ) from exc
    if (
        not trusted_root.is_dir()
        or requested_root.is_symlink()
        or not approved.is_file()
        or requested_approved.is_symlink()
        or not approved.is_relative_to(trusted_root)
        or any(
            path.is_symlink()
            for path in (
                trusted_root.joinpath(*relative_path.parts[:index])
                for index in range(1, len(relative_path.parts) + 1)
            )
        )
    ):
        raise Lfm2CpcfCatalogError(
            "Approved V5 GPU parity artifact escaped its repository root."
        )
    if not isinstance(value, Mapping):
        raise Lfm2CpcfCatalogError(
            "V5 proven GPU parity authority must be a sealed file reference."
        )
    reference = deepcopy(dict(value))
    if set(reference) != {
        "path",
        "file_sha256",
        "receipt_sha256",
        "source_sha256",
        "scheduler_policy",
        "target_dtype",
        "real_two_gpu_parity_passed",
    }:
        raise Lfm2CpcfCatalogError(
            "V5 proven GPU parity file reference is incomplete or expanded."
        )
    raw_path = reference.get("path")
    if not isinstance(raw_path, str) or not raw_path or "://" in raw_path:
        raise Lfm2CpcfCatalogError(
            "V5 GPU parity receipt path must be an absolute local path."
        )
    requested = Path(raw_path).expanduser()
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise Lfm2CpcfCatalogError(
            "V5 GPU parity receipt path does not exist."
        ) from exc
    if (
        not requested.is_absolute()
        or requested.is_symlink()
        or not resolved.is_file()
        or str(resolved) != raw_path
        or resolved != approved
    ):
        raise Lfm2CpcfCatalogError(
            "V5 GPU parity receipt path is not canonical, approved, or safe."
        )
    try:
        raw = resolved.read_bytes()
        document = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CpcfCatalogError(
            "V5 GPU parity receipt is unreadable or not canonical JSON."
        ) from exc
    if (
        not isinstance(document, Mapping)
        or set(document) != _V5_GPU_PARITY_RECEIPT_FIELDS
        or raw != _canonical_json(document) + b"\n"
    ):
        raise Lfm2CpcfCatalogError(
            "V5 GPU parity receipt fields or canonical file encoding differ."
        )
    receipt = deepcopy(dict(document))
    receipt_sha256 = _digest(
        receipt.pop("receipt_sha256", None),
        "v5 GPU parity receipt_sha256",
    )
    file_sha256 = hashlib.sha256(raw).hexdigest()
    sources = receipt.get("source_sha256")
    devices = receipt.get("devices")
    backend = receipt.get("backend_report")
    if not isinstance(backend, Mapping):
        raise Lfm2CpcfCatalogError(
            "V5 GPU parity receipt has no sealed backend report."
        )
    backend_body = deepcopy(dict(backend))
    backend_sha256 = _digest(
        backend_body.pop("report_sha256", None),
        "v5 GPU parity backend report_sha256",
    )
    execution = backend.get("execution_report")
    if not isinstance(execution, Mapping):
        raise Lfm2CpcfCatalogError(
            "V5 GPU parity receipt has no sealed execution report."
        )
    execution_body = deepcopy(dict(execution))
    execution_sha256 = _digest(
        execution_body.pop("report_sha256", None),
        "v5 GPU parity execution report_sha256",
    )
    plan = execution.get("plan")
    if not isinstance(plan, Mapping):
        raise Lfm2CpcfCatalogError(
            "V5 GPU parity receipt has no sealed scheduler plan."
        )
    plan_body = deepcopy(dict(plan))
    plan_sha256 = _digest(
        plan_body.pop("plan_sha256", None),
        "v5 GPU parity plan_sha256",
    )
    expected_policy = deepcopy(dict(expected_scheduler_policy))
    live_sources = dict(_v5_gpu_parity_source_sha256())
    metric_delta = receipt.get("maximum_metric_delta")
    allowed_delta = receipt.get("maximum_allowed_metric_delta")
    scheduled = execution.get("scheduled_records")
    parity_reports = execution.get("parity_reports")
    compute_devices = backend.get("compute_device_by_work_id")
    if (
        canonical_sha256(receipt) != receipt_sha256
        or canonical_sha256(backend_body) != backend_sha256
        or canonical_sha256(execution_body) != execution_sha256
        or canonical_sha256(plan_body) != plan_sha256
        or receipt.get("schema") != LFM2_CPCF_V5_GPU_PARITY_RECEIPT_SCHEMA
        or receipt.get("status") != "passed"
        or receipt.get("target_dtype") != "bfloat16"
        or receipt.get("candidate_state_sha256_equal") is not True
        or receipt.get("closed_form_solve_math_changed") is not False
        or any(
            receipt.get(name) != 0
            for name in (
                "training_steps",
                "backward_calls",
                "optimizer_steps",
                "parameters_pruned",
                "parameters_quantized",
            )
        )
        or receipt.get("candidate_count") != 4
        or receipt.get("case_count") != 3
        or not isinstance(metric_delta, (int, float))
        or isinstance(metric_delta, bool)
        or not math.isfinite(float(metric_delta))
        or not isinstance(allowed_delta, (int, float))
        or isinstance(allowed_delta, bool)
        or not math.isfinite(float(allowed_delta))
        or float(allowed_delta) != _V5_GPU_PARITY_FIXED_MAXIMUM_METRIC_DELTA
        or float(metric_delta) > float(allowed_delta)
        or not isinstance(sources, Mapping)
        or set(sources) != _V5_GPU_PARITY_SOURCE_FILENAMES
        or dict(sources) != live_sources
        or backend.get("schema") != LFM2_CPCF_CANDIDATE_REPLAY_BACKEND_SCHEMA
        or backend.get("kind") != "authenticated_two_cuda_scheduler"
        or backend.get("representative_cpu_cuda_parity_passed") is not True
        or backend.get("canonical_candidate_order_restored") is not True
        or backend.get("closed_form_solve_math_changed") is not False
        or execution.get("schema") != DYNAMIC_REPLAY_EXECUTION_SCHEMA
        or execution.get("plan_sha256") != plan_sha256
        or execution.get("parity_passed_before_scheduled_work") is not True
        or execution.get("backend_receipts_reauthenticated") is not True
        or execution.get("installed_state_reauthenticated") is not True
        or execution.get("cpu_reference_state_reauthenticated") is not True
        or execution.get("canonical_result_order_restored") is not True
        or execution.get("dynamic_work_stealing") is not False
        or execution.get("folding_math_changed") is not False
        or plan.get("policy") != expected_policy
        or plan.get("dynamic_work_stealing") is not False
        or plan.get("folding_math_changed") is not False
        or not isinstance(scheduled, list)
        or not scheduled
        or {row.get("device_index") for row in scheduled} != {0, 1}
        or not isinstance(parity_reports, list)
        or {row.get("device_index") for row in parity_reports} != {0, 1}
        or any(row.get("passed") is not True for row in parity_reports)
        or not isinstance(compute_devices, Mapping)
        or not compute_devices
        or set(compute_devices.values()) != {"cuda:0", "cuda:1"}
    ):
        raise Lfm2CpcfCatalogError(
            "V5 GPU parity receipt failed its schema, seal, BF16, state, "
            "scheduler, source, execution, or zero-operation contract."
        )
    if (
        not isinstance(devices, list)
        or len(devices) != 2
        or [row.get("device_index") for row in devices] != [0, 1]
        or any(
            not isinstance(row, Mapping)
            or row.get("backend") != "cuda"
            or row.get("device_name") != "NVIDIA GeForce RTX 3090"
            or "+cpu" in str(row.get("torch_version"))
            or row.get("deterministic_algorithms_enabled") is not True
            or row.get("deterministic_debug_mode") != 2
            or row.get("cudnn_deterministic") is not True
            or row.get("cudnn_benchmark") is not False
            or row.get("cudnn_allow_tf32") is not False
            or row.get("cuda_matmul_allow_tf32") is not False
            or row.get("cuda_matmul_allow_bf16_reduced_precision_reduction")
            is not False
            or row.get("cuda_matmul_allow_fp16_reduced_precision_reduction")
            is not False
            or row.get("float32_matmul_precision") != "highest"
            or row.get("cublas_workspace_config") != ":4096:8"
            or canonical_sha256(
                {name: child for name, child in row.items() if name != "receipt_sha256"}
            )
            != row.get("receipt_sha256")
            for row in devices
        )
        or len({row.get("device_uuid") for row in devices}) != 2
        or len(
            {
                (
                    row.get("pci_domain_id"),
                    row.get("pci_bus_id"),
                    row.get("pci_device_id"),
                )
                for row in devices
            }
        )
        != 2
    ):
        raise Lfm2CpcfCatalogError(
            "V5 GPU parity receipt lacks two distinct deterministic RTX "
            "3090 device receipts."
        )
    expected_reference = {
        "path": str(resolved),
        "file_sha256": file_sha256,
        "receipt_sha256": receipt_sha256,
        "source_sha256": live_sources,
        "scheduler_policy": expected_policy,
        "target_dtype": "bfloat16",
        "real_two_gpu_parity_passed": True,
    }
    if (
        file_sha256 != authority.file_sha256
        or receipt_sha256 != authority.receipt_sha256
        or reference != expected_reference
    ):
        raise Lfm2CpcfCatalogError(
            "V5 GPU parity file or receipt differs from its exact approved "
            "source-controlled authority."
        )
    return MappingProxyType(expected_reference)


def _build_v5_gpu_parity_production_loader(
    authority: _V5GpuParityAuthority | None,
    repository_root: Path,
) -> Callable[..., Mapping[str, Any]]:
    """Close production authority over immutable import-time values."""

    core_authenticator = _authenticate_v5_gpu_parity_receipt_against_authority

    def authenticate(
        value: Any,
        *,
        expected_scheduler_policy: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if authority is None:
            raise Lfm2CpcfCatalogError(
                "No exact V5 two-RTX-3090 parity receipt is currently "
                "approved; production remains fail-closed until the verifier "
                "is rerun and the source-controlled authority constant is "
                "updated."
            )
        return core_authenticator(
            value,
            expected_scheduler_policy=expected_scheduler_policy,
            authority=authority,
            repository_root=repository_root,
        )

    return authenticate


_authenticate_v5_gpu_parity_receipt_file = _build_v5_gpu_parity_production_loader(
    _PRODUCTION_V5_GPU_PARITY_AUTHORITY,
    _V5_REPOSITORY_ROOT,
)


def _test_only_authenticate_v5_gpu_parity_receipt_file(
    value: Any,
    *,
    expected_scheduler_policy: Mapping[str, Any],
    approved_repository_root: Path,
    approved_repository_relative_path: str,
    approved_file_sha256: str,
    approved_receipt_sha256: str,
) -> Mapping[str, Any]:
    """Internal seam for semantic tests; never used by production wiring."""

    authority = _V5GpuParityAuthority(
        repository_relative_path=approved_repository_relative_path,
        file_sha256=approved_file_sha256,
        receipt_sha256=approved_receipt_sha256,
    )
    return _authenticate_v5_gpu_parity_receipt_against_authority(
        value,
        expected_scheduler_policy=expected_scheduler_policy,
        authority=authority,
        repository_root=approved_repository_root,
    )


def _authenticated_v5_gpu_backend_report(
    value: Mapping[str, Any],
    evaluator: AuthenticatedTwoGpuLfm2CpcfCandidateReplayEvaluator,
    *,
    _runtime_authenticator: Callable[..., Mapping[str, Any]] = (
        _authenticate_v5_runtime_objects
    ),
    _parity_loader: Callable[..., Mapping[str, Any]] = (
        _authenticate_v5_gpu_parity_receipt_file
    ),
    _evaluator_class: type = _AUTHENTIC_TWO_GPU_EVALUATOR_CLASS,
    _scheduler_class: type = _AUTHENTIC_REPLAY_SCHEDULER_CLASS,
    _scheduler_policy_class: type = (_AUTHENTIC_REPLAY_SCHEDULER_POLICY_CLASS),
    _state_probes_class: type = _AUTHENTIC_REPLAY_STATE_PROBES_CLASS,
    _host_factory_class: type = _AUTHENTIC_PINNED_HOST_FACTORY_CLASS,
    _cuda_worker: Callable[..., Any] = _AUTHENTIC_CUDA_REPLAY_WORKER,
    _cpu_worker: Callable[..., Any] = _AUTHENTIC_CPU_PARITY_WORKER,
    _device_probe: Callable[..., Any] = _AUTHENTIC_DEVICE_RECEIPT_PROBE,
    _evaluator_evaluate: Callable[..., Any] = (
        _AUTHENTIC_TWO_GPU_EVALUATOR_CLASS.evaluate
    ),
    _scheduler_authenticate_runtime: Callable[..., Mapping[str, Any]] = (
        _AUTHENTIC_REPLAY_SCHEDULER_CLASS.authenticate_runtime
    ),
    _scheduler_execute: Callable[..., Any] = (
        _AUTHENTIC_REPLAY_SCHEDULER_CLASS.execute
    ),
    _scheduler_policy_as_dict: Callable[..., dict[str, Any]] = (
        _AUTHENTIC_REPLAY_SCHEDULER_POLICY_CLASS.as_dict
    ),
) -> Mapping[str, Any]:
    """Bind exact live CUDA execution to its prior parity envelope."""

    _runtime_authenticator()
    if (
        type(evaluator) is not _evaluator_class
        or type(evaluator).evaluate is not _evaluator_evaluate
    ):
        raise Lfm2CpcfCatalogError(
            "V5 production candidate replay requires the exact authenticated "
            "two-GPU evaluator type."
        )
    scheduler = evaluator.scheduler
    if (
        type(scheduler) is not _scheduler_class
        or type(scheduler.policy) is not _scheduler_policy_class
        or type(scheduler).authenticate_runtime is not _scheduler_authenticate_runtime
        or type(scheduler).execute is not _scheduler_execute
    ):
        raise Lfm2CpcfCatalogError(
            "V5 production candidate replay requires the exact deterministic "
            "replay scheduler."
        )
    state_owner = getattr(scheduler.installed_state_probe, "__self__", None)
    cpu_state_owner = getattr(
        scheduler.cpu_installed_state_probe,
        "__self__",
        None,
    )
    if (
        scheduler.gpu_worker is not _cuda_worker
        or scheduler.cpu_parity_worker is not _cpu_worker
        or scheduler.device_receipt_probe is not _device_probe
        or type(state_owner) is not _state_probes_class
        or cpu_state_owner is not state_owner
        or getattr(
            scheduler.installed_state_probe,
            "__func__",
            None,
        )
        is not _state_probes_class.installed_state_probe
        or getattr(
            scheduler.cpu_installed_state_probe,
            "__func__",
            None,
        )
        is not _state_probes_class.cpu_installed_state_probe
        or type(scheduler.host_buffer_factory) is not _host_factory_class
        or type(scheduler.host_buffer_factory).__call__
        is not _host_factory_class.__call__
    ):
        raise Lfm2CpcfCatalogError(
            "V5 production GPU scheduler callables, live state probes, or "
            "pinned-host factory are not the exact source-bound runtime."
        )
    try:
        live_runtime = dict(_scheduler_authenticate_runtime(scheduler))
    except BaseException as exc:
        raise Lfm2CpcfCatalogError(
            "V5 production could not freshly authenticate two live CUDA "
            "devices and their retained replay state."
        ) from exc
    report = deepcopy(dict(value))
    report_sha256 = _digest(
        report.pop("report_sha256", None),
        "v5 GPU backend report_sha256",
    )
    replicas = tuple(scheduler.replicas)
    device_indices = [int(replica.device.device_index) for replica in replicas]
    parity = report.get("proven_parity_receipt")
    scheduler_policy = _scheduler_policy_as_dict(scheduler.policy)
    authenticated_parity = _parity_loader(
        parity,
        expected_scheduler_policy=scheduler_policy,
    )
    if (
        canonical_sha256(report) != report_sha256
        or report.get("schema") != LFM2_CPCF_V5_GPU_REPLAY_SCHEMA
        or len(replicas) != 2
        or len(set(device_indices)) != 2
        or report.get("logical_device_indices") != device_indices
        or report.get("replicas") != [replica.as_dict() for replica in replicas]
        or report.get("scheduler_policy") != scheduler_policy
        or report.get("runtime_authentication") != live_runtime
        or report.get("runtime_authentication_sha256") != live_runtime["report_sha256"]
        or report.get("homogeneous_authenticated_receipts") is not True
        or report.get("candidate_payload_staging")
        != "bounded_double_buffered_pinned_system_memory_to_cuda"
        or parity != dict(authenticated_parity)
        or any(
            replica.device.device_name != scheduler.policy.expected_device_name
            or "+cpu" in replica.device.torch_version
            or not replica.device.cuda_runtime_version
            for replica in replicas
        )
    ):
        raise Lfm2CpcfCatalogError(
            "V5 GPU evaluator differs from its sealed two-device parity, "
            "scheduler, replica, or pinned-host staging envelope."
        )
    return MappingProxyType({**report, "report_sha256": report_sha256})


_AUTHENTIC_V5_GPU_BACKEND_REPORT_AUTHENTICATOR = _authenticated_v5_gpu_backend_report


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5SelectiveBackend:
    """Source-bound bounded-residency v5 compiler and real GPU replay seam.

    The full compiler report intentionally is not preregistered byte-for-byte:
    it contains sampled process-RSS values and therefore changes between
    otherwise deterministic reruns.  Stable compiler, cache, catalog, policy,
    phase, numerical, selector, and GPU-parity identities are pinned here;
    the reviewed gate independently pins the exact resulting child state and
    all 64 whole-output responses.
    """

    cache: Lfm2CircuitTraceCache
    runtime_policy: Lfm2CpcfV5RuntimePolicy
    candidate_replay_evaluator: AuthenticatedTwoGpuLfm2CpcfCandidateReplayEvaluator
    candidate_selector: Lfm2CpcfV5EndpointFirstCandidateSelector
    numerical_frontier_adapter: Lfm2CpcfShortConvV5NumericalFrontierAdapter
    gpu_backend_report: Mapping[str, Any]
    expected_selective_catalog_sha256: str
    expected_selective_source_sha256: str
    expected_selective_provider_sha256: str
    expected_layer_count: int
    expected_phase_sha256: tuple[str, ...]
    compiler_authentication: Mapping[str, Any] = field(init=False)
    backend_fingerprint: str = field(init=False)

    def __post_init__(
        self,
        _gpu_backend_authenticator: Callable[
            [Mapping[str, Any], Any],
            Mapping[str, Any],
        ] = _AUTHENTIC_V5_GPU_BACKEND_REPORT_AUTHENTICATOR,
    ) -> None:
        if type(self.cache) is not Lfm2CircuitTraceCache:
            raise Lfm2CpcfCatalogError(
                "V5 cache must be the exact immutable LFM2 trace-cache type."
            )
        manifest = self.cache.manifest
        manifest_sha256 = _digest(
            manifest.pop("manifest_sha256", None),
            "v5 cache manifest_sha256",
        )
        if (
            canonical_sha256(manifest) != manifest_sha256
            or manifest.get("cache_key") != self.cache.cache_key
        ):
            raise Lfm2CpcfCatalogError("V5 trace-cache manifest seal is invalid.")
        if type(self.runtime_policy) is not Lfm2CpcfV5RuntimePolicy:
            raise Lfm2CpcfCatalogError(
                "runtime_policy must be the exact v5 runtime-policy type."
            )
        model_policy = self.runtime_policy.model_fold_policy
        if (
            self.runtime_policy.require_short_conv_v5_endpoint_first_selector
            is not True
            or self.runtime_policy.require_short_conv_v5_real_numerical_frontier_adapter
            is not True
            or model_policy.require_promotion_ratio is not True
            or float(model_policy.promotion_compression_ratio) < 100.0
            or self.runtime_policy.full_phase_tree_logical_bytes
            <= self.runtime_policy.maximum_resident_evidence_bytes
        ):
            raise Lfm2CpcfCatalogError(
                "V5 production requires the real numerical frontier, "
                "endpoint-first selection, a preregistered >=100x parameter "
                "promotion ratio, and bounded residency below the full phase "
                "tree."
            )
        if type(self.candidate_selector) is not (
            Lfm2CpcfV5EndpointFirstCandidateSelector
        ):
            raise Lfm2CpcfCatalogError(
                "V5 production requires the exact endpoint-first selector."
            )
        if type(self.numerical_frontier_adapter) is not (
            Lfm2CpcfShortConvV5NumericalFrontierAdapter
        ):
            raise Lfm2CpcfCatalogError(
                "V5 production requires the concrete real numerical frontier adapter."
            )
        numerical_contract = _real_v5_numerical_adapter_contract(
            self.numerical_frontier_adapter
        )
        if not isinstance(numerical_contract, Mapping):
            raise Lfm2CpcfCatalogError(
                "V5 numerical adapter failed its authenticated policy contract."
            )
        gpu_report = _gpu_backend_authenticator(
            self.gpu_backend_report,
            self.candidate_replay_evaluator,
        )
        object.__setattr__(
            self,
            "gpu_backend_report",
            MappingProxyType(deepcopy(dict(gpu_report))),
        )
        for name in (
            "expected_selective_catalog_sha256",
            "expected_selective_source_sha256",
            "expected_selective_provider_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), name),
            )
        if (
            isinstance(self.expected_layer_count, bool)
            or not isinstance(self.expected_layer_count, int)
            or self.expected_layer_count <= 0
        ):
            raise Lfm2CpcfCatalogError(
                "expected_layer_count must be a positive integer."
            )
        phases = tuple(
            _digest(value, "expected_phase_sha256")
            for value in self.expected_phase_sha256
        )
        if not phases or len(set(phases)) != len(phases):
            raise Lfm2CpcfCatalogError(
                "V5 expected phase seals must be nonempty, ordered, and unique."
            )
        object.__setattr__(self, "expected_phase_sha256", phases)
        compiler_callable = _authenticate_v5_compiler_callable()
        object.__setattr__(
            self,
            "compiler_authentication",
            MappingProxyType(deepcopy(dict(compiler_callable))),
        )
        compiler = dict(lfm2_cpcf_v5_compiler_implementation())
        implementation_sha256 = _digest(
            compiler.get("implementation_sha256"),
            "v5 compiler implementation_sha256",
        )
        compiler_body = dict(compiler)
        compiler_body.pop("implementation_sha256")
        if canonical_sha256(compiler_body) != implementation_sha256:
            raise Lfm2CpcfCatalogError(
                "Live v5 compiler implementation seal is invalid."
            )
        object.__setattr__(
            self,
            "backend_fingerprint",
            canonical_sha256(
                {
                    "schema": LFM2_CPCF_V5_BACKEND_SCHEMA,
                    "cache_key": self.cache.cache_key,
                    "cache_manifest_sha256": manifest_sha256,
                    "cache_parent_fingerprint": (self.cache.parent_fingerprint),
                    "compiler_implementation_sha256": (implementation_sha256),
                    "compiler_callable_authentication_sha256": (
                        compiler_callable["authentication_sha256"]
                    ),
                    "runtime_policy": self.runtime_policy.as_dict(),
                    "selective_catalog_sha256": (
                        self.expected_selective_catalog_sha256
                    ),
                    "selective_source_sha256": (self.expected_selective_source_sha256),
                    "selective_provider_sha256": (
                        self.expected_selective_provider_sha256
                    ),
                    "expected_layer_count": self.expected_layer_count,
                    "ordered_phase_sha256": list(phases),
                    "gpu_backend_report_sha256": gpu_report["report_sha256"],
                    "gpu_parity_receipt_sha256": gpu_report["proven_parity_receipt"][
                        "receipt_sha256"
                    ],
                    "gpu_parity_file_sha256": gpu_report["proven_parity_receipt"][
                        "file_sha256"
                    ],
                    "numerical_adapter_contract_sha256": (
                        numerical_contract["contract_sha256"]
                    ),
                    "candidate_selector_type": (
                        f"{type(self.candidate_selector).__module__}."
                        f"{type(self.candidate_selector).__qualname__}"
                    ),
                    "selective_cpu_residency": True,
                    "bounded_pinned_host_to_cuda_staging": True,
                    "training_used": False,
                    "pruning_used": False,
                    "quantization_used": False,
                }
            ),
        )

    def verify(
        self,
        _gpu_backend_authenticator: Callable[
            [Mapping[str, Any], Any],
            Mapping[str, Any],
        ] = _AUTHENTIC_V5_GPU_BACKEND_REPORT_AUTHENTICATOR,
    ) -> Lfm2CpcfV5SelectiveBackend:
        """Reauthenticate every mutable nested object before use."""

        gpu_report = _gpu_backend_authenticator(
            self.gpu_backend_report,
            self.candidate_replay_evaluator,
        )
        if (
            type(self.candidate_selector)
            is not Lfm2CpcfV5EndpointFirstCandidateSelector
            or type(self.numerical_frontier_adapter)
            is not Lfm2CpcfShortConvV5NumericalFrontierAdapter
            or type(self.runtime_policy) is not Lfm2CpcfV5RuntimePolicy
        ):
            raise Lfm2CpcfCatalogError(
                "V5 backend component types changed after construction."
            )
        numerical_contract = _real_v5_numerical_adapter_contract(
            self.numerical_frontier_adapter
        )
        compiler_callable = _authenticate_v5_compiler_callable()
        compiler = dict(lfm2_cpcf_v5_compiler_implementation())
        implementation_sha256 = _digest(
            compiler.pop("implementation_sha256", None),
            "v5 compiler implementation_sha256",
        )
        if (
            not isinstance(numerical_contract, Mapping)
            or canonical_sha256(compiler) != implementation_sha256
            or dict(compiler_callable) != dict(self.compiler_authentication)
        ):
            raise Lfm2CpcfCatalogError(
                "V5 compiler or numerical implementation authentication changed."
            )
        manifest = self.cache.manifest
        manifest_sha256 = _digest(
            manifest.pop("manifest_sha256", None),
            "v5 cache manifest_sha256",
        )
        observed_fingerprint = canonical_sha256(
            {
                "schema": LFM2_CPCF_V5_BACKEND_SCHEMA,
                "cache_key": self.cache.cache_key,
                "cache_manifest_sha256": manifest_sha256,
                "cache_parent_fingerprint": self.cache.parent_fingerprint,
                "compiler_implementation_sha256": implementation_sha256,
                "compiler_callable_authentication_sha256": (
                    compiler_callable["authentication_sha256"]
                ),
                "runtime_policy": self.runtime_policy.as_dict(),
                "selective_catalog_sha256": (self.expected_selective_catalog_sha256),
                "selective_source_sha256": (self.expected_selective_source_sha256),
                "selective_provider_sha256": (self.expected_selective_provider_sha256),
                "expected_layer_count": self.expected_layer_count,
                "ordered_phase_sha256": list(self.expected_phase_sha256),
                "gpu_backend_report_sha256": gpu_report["report_sha256"],
                "gpu_parity_receipt_sha256": gpu_report["proven_parity_receipt"][
                    "receipt_sha256"
                ],
                "gpu_parity_file_sha256": gpu_report["proven_parity_receipt"][
                    "file_sha256"
                ],
                "numerical_adapter_contract_sha256": (
                    numerical_contract["contract_sha256"]
                ),
                "candidate_selector_type": (
                    f"{type(self.candidate_selector).__module__}."
                    f"{type(self.candidate_selector).__qualname__}"
                ),
                "selective_cpu_residency": True,
                "bounded_pinned_host_to_cuda_staging": True,
                "training_used": False,
                "pruning_used": False,
                "quantization_used": False,
            }
        )
        if (
            canonical_sha256(manifest) != manifest_sha256
            or observed_fingerprint != self.backend_fingerprint
        ):
            raise Lfm2CpcfCatalogError(
                "V5 selective backend changed after construction."
            )
        return self

    def authenticated_compiler(self) -> Callable[..., Lfm2CpcfCompiledModelFold]:
        """Return only the exact compiler callable pinned at construction."""

        self.verify()
        _authenticate_v5_compiler_callable()
        return _AUTHENTIC_V5_COMPILER_CALLABLE

    def open_provider(self) -> Lfm2CpcfSelectiveEvidenceProvider:
        """Open and authenticate one bounded, one-use selective provider."""

        self.verify()
        provider = build_lfm2_cpcf_selective_evidence_provider(self.cache)
        try:
            catalog = provider.catalog
            full_bytes = int(catalog.final_norm_logical_bytes) + sum(
                int(reference.logical_bytes)
                for reference in catalog.dynamic_catalog.references
            )
            if (
                catalog.cache_key != self.cache.cache_key
                or catalog.catalog_sha256 != self.expected_selective_catalog_sha256
                or catalog.source_sha256 != self.expected_selective_source_sha256
                or catalog.provider_sha256 != self.expected_selective_provider_sha256
                or catalog.layer_count != self.expected_layer_count
                or full_bytes != self.runtime_policy.full_phase_tree_logical_bytes
            ):
                raise Lfm2CpcfCatalogError(
                    "Live selective provider differs from the preregistered "
                    "cache catalog, source, implementation, or logical byte "
                    "tree."
                )
        except BaseException:
            provider.close()
            raise
        return provider


@dataclass(frozen=True, slots=True)
class Lfm2CpcfStackResources:
    """Reusable, already-materialized CPCF evidence and publication callbacks."""

    phases: tuple[Lfm2CpcfTargetPhase, ...]
    trace_adapter: Lfm2StrictTokenizerBridgeTraceAdapter
    coverage_binding: Lfm2CpcfCachePhaseCoverageBinding
    parent_graph_fingerprint: str
    source_tied_table: Any
    policy: Lfm2CpcfModelFoldPolicy
    external_behavior_contract: Lfm2CpcfExternalBehaviorGateContract
    expected_full_reviewed_evidence_fingerprint: str
    final_evaluator: Lfm2CpcfReviewedPromotionGate
    model_saver: Callable[[Any, Path], None]
    model_loader: Callable[[Path], Any]
    fresh_reload_evaluator: Lfm2CpcfReviewedPromotionGate
    saver_id: str
    loader_id: str
    snapshot_config: DynamicFoldSnapshotConfig | None = None
    resource_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        phases = tuple(self.phases)
        if not phases or any(
            not isinstance(value, Lfm2CpcfTargetPhase) for value in phases
        ):
            raise Lfm2CpcfCatalogError("phases must contain typed CPCF target phases.")
        object.__setattr__(self, "phases", phases)
        if not isinstance(
            self.trace_adapter,
            Lfm2StrictTokenizerBridgeTraceAdapter,
        ):
            raise Lfm2CpcfCatalogError(
                "trace_adapter must be a strict tokenizer bridge adapter."
            )
        self.trace_adapter.verify()
        object.__setattr__(
            self,
            "parent_graph_fingerprint",
            _digest(
                self.parent_graph_fingerprint,
                "parent_graph_fingerprint",
            ),
        )
        if not isinstance(self.policy, Lfm2CpcfModelFoldPolicy):
            raise Lfm2CpcfCatalogError("policy must be Lfm2CpcfModelFoldPolicy.")
        if not isinstance(
            self.external_behavior_contract,
            Lfm2CpcfExternalBehaviorGateContract,
        ):
            raise Lfm2CpcfCatalogError("external_behavior_contract has the wrong type.")
        object.__setattr__(
            self,
            "expected_full_reviewed_evidence_fingerprint",
            _digest(
                self.expected_full_reviewed_evidence_fingerprint,
                "expected_full_reviewed_evidence_fingerprint",
            ),
        )
        for name in ("model_saver", "model_loader"):
            if not callable(getattr(self, name)):
                raise Lfm2CpcfCatalogError(
                    f"CPCF stack resource {name} must be callable."
                )
        if (
            type(self.final_evaluator) is not Lfm2CpcfReviewedPromotionGate
            or self.fresh_reload_evaluator is not self.final_evaluator
            or not callable(getattr(self.final_evaluator, "evaluate", None))
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF final and fresh-reload evaluation must use one exact "
                "typed reviewed-promotion gate."
            )
        final_binding = _strict_behavior_binding(
            self.final_evaluator,
            self.external_behavior_contract,
            name="final_evaluator",
        )
        reload_binding = _strict_behavior_binding(
            self.fresh_reload_evaluator,
            self.external_behavior_contract,
            name="fresh_reload_evaluator",
        )
        if final_binding.manifest_fingerprint != reload_binding.manifest_fingerprint:
            raise Lfm2CpcfCatalogError(
                "CPCF final and fresh-reload behavior manifests differ."
            )
        reviewed = self.final_evaluator.evidence
        if (
            reviewed.compiled_model_fold_schema != LFM2_CPCF_MODEL_FOLD_SCHEMA
            or reviewed.full_reviewed_evidence_fingerprint
            != self.expected_full_reviewed_evidence_fingerprint
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF reviewed promotion gate differs from the independently "
                "pinned full reviewed-evidence fingerprint."
            )
        if (
            reviewed.bridge_artifact_sha256 != self.trace_adapter.bridge_artifact_sha256
            or reviewed.bridge_profile_sha256
            != self.trace_adapter.transport_profile_fingerprint
            or reviewed.target_tokenizer_semantics_sha256
            != self.trace_adapter.target_tokenizer_semantics_sha256
            or reviewed.candidate_tokenizer_fingerprint
            != self.trace_adapter.target_tokenizer_package_fingerprint
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF reviewed promotion evidence is not bound to the exact "
                "strict tokenizer bridge artifact, transport profile, and "
                "target tokenizer package/semantics used by the trace "
                "adapter."
            )
        coverage_fields = _authenticate_cpcf_resource_coverage(
            coverage_binding=self.coverage_binding,
            phases=phases,
            trace_adapter=self.trace_adapter,
            parent_graph_fingerprint=self.parent_graph_fingerprint,
            policy=self.policy,
            reviewed=reviewed,
        )
        object.__setattr__(self, "saver_id", _text(self.saver_id, "saver_id"))
        object.__setattr__(self, "loader_id", _text(self.loader_id, "loader_id"))
        if self.snapshot_config is not None and not isinstance(
            self.snapshot_config,
            DynamicFoldSnapshotConfig,
        ):
            raise Lfm2CpcfCatalogError(
                "snapshot_config must be DynamicFoldSnapshotConfig or None."
            )
        table = self.source_tied_table
        torch = _torch()
        if (
            not isinstance(table, torch.Tensor)
            or table.layout != torch.strided
            or table.device.type == "meta"
            or table.ndim != 2
            or table.numel() == 0
            or not torch.is_floating_point(table)
            or table.requires_grad
            or table.grad_fn is not None
            or not bool(torch.isfinite(table.detach()).all().item())
        ):
            raise Lfm2CpcfCatalogError(
                "source_tied_table must be a finite detached dense rank-two "
                "floating tensor."
            )
        phase_cache_keys = {phase.parent_evidence.cache_key for phase in phases}
        if len(phase_cache_keys) != 1:
            raise Lfm2CpcfCatalogError(
                "CPCF stack resources must bind one parent cache."
            )
        resource_body = {
            "schema": LFM2_CPCF_RESOURCE_SCHEMA,
            "cache_key": _digest(
                phase_cache_keys.pop(),
                "resource cache_key",
            ),
            "ordered_phase_sha256": [
                _phase_metadata(phase)["phase_sha256"] for phase in phases
            ],
            "ordered_parent_evidence_sha256": [
                _digest(
                    phase.parent_evidence.evidence_sha256,
                    "phase parent_evidence.evidence_sha256",
                )
                for phase in phases
            ],
            "trace_adapter_sha256": _digest(
                self.trace_adapter.evidence_sha256,
                "trace_adapter.evidence_sha256",
            ),
            "parent_graph_fingerprint": self.parent_graph_fingerprint,
            "bridge_artifact_sha256": _digest(
                self.trace_adapter.bridge_artifact_sha256,
                "trace_adapter.bridge_artifact_sha256",
            ),
            "transport_profile_sha256": _digest(
                self.trace_adapter.transport_profile_fingerprint,
                "trace_adapter.transport_profile_fingerprint",
            ),
            "source_tied_table_sha256": _tensor_sha256(
                table,
                "source_tied_table",
            ),
            "policy_sha256": canonical_sha256(self.policy.as_dict()),
            **coverage_fields,
            "external_behavior_contract_sha256": (
                self.external_behavior_contract.contract_sha256
            ),
            "behavior_gate_manifest_fingerprint": (final_binding.manifest_fingerprint),
            "reviewed_bridge_artifact_sha256": (reviewed.bridge_artifact_sha256),
            "reviewed_compiled_model_fold_schema": (
                reviewed.compiled_model_fold_schema
            ),
            "reviewed_bridge_profile_sha256": (reviewed.bridge_profile_sha256),
            "reviewed_target_tokenizer_semantics_sha256": (
                reviewed.target_tokenizer_semantics_sha256
            ),
            "reviewed_candidate_tokenizer_fingerprint": (
                reviewed.candidate_tokenizer_fingerprint
            ),
            "expected_full_reviewed_evidence_fingerprint": (
                self.expected_full_reviewed_evidence_fingerprint
            ),
            "saver_id": self.saver_id,
            "loader_id": self.loader_id,
            "snapshot_config": (
                None if self.snapshot_config is None else self.snapshot_config.as_dict()
            ),
            "target_blank_fingerprint_bound": False,
        }
        object.__setattr__(
            self,
            "resource_fingerprint",
            canonical_sha256(resource_body),
        )


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5SelectiveStackResources:
    """Production callbacks and reviewed behavior authority for v5."""

    backend: Lfm2CpcfV5SelectiveBackend
    trace_adapter: Lfm2StrictTokenizerBridgeTraceAdapter
    parent_graph_fingerprint: str
    source_tied_table: Any
    external_behavior_contract: Lfm2CpcfExternalBehaviorGateContract
    expected_full_reviewed_evidence_fingerprint: str
    final_evaluator: Lfm2CpcfReviewedPromotionGate
    model_saver: Callable[[Any, Path], None]
    model_loader: Callable[[Path], Any]
    fresh_reload_evaluator: Lfm2CpcfReviewedPromotionGate
    saver_id: str
    loader_id: str
    resume_coordinator: Lfm2CpcfV5ResumeCoordinator | None = None
    resume_state: Lfm2CpcfV5ResumeState | None = None
    resume_admission_sha256: str | None = None
    snapshot_config: DynamicFoldSnapshotConfig | None = None
    resource_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.backend) is not Lfm2CpcfV5SelectiveBackend:
            raise Lfm2CpcfCatalogError(
                "backend must be the exact selective v5 backend type."
            )
        self.backend.verify()
        if type(self.trace_adapter) is not (Lfm2StrictTokenizerBridgeTraceAdapter):
            raise Lfm2CpcfCatalogError(
                "trace_adapter must be the strict tokenizer bridge adapter."
            )
        trace = self.trace_adapter.verify()
        object.__setattr__(
            self,
            "parent_graph_fingerprint",
            _digest(
                self.parent_graph_fingerprint,
                "parent_graph_fingerprint",
            ),
        )
        if type(self.external_behavior_contract) is not (
            Lfm2CpcfExternalBehaviorGateContract
        ):
            raise Lfm2CpcfCatalogError("external_behavior_contract has the wrong type.")
        object.__setattr__(
            self,
            "expected_full_reviewed_evidence_fingerprint",
            _digest(
                self.expected_full_reviewed_evidence_fingerprint,
                "expected_full_reviewed_evidence_fingerprint",
            ),
        )
        for name in ("model_saver", "model_loader"):
            if not callable(getattr(self, name)):
                raise Lfm2CpcfCatalogError(
                    f"V5 CPCF stack resource {name} must be callable."
                )
        if (
            type(self.final_evaluator) is not Lfm2CpcfReviewedPromotionGate
            or self.fresh_reload_evaluator is not self.final_evaluator
        ):
            raise Lfm2CpcfCatalogError(
                "V5 final and fresh-reload evaluation must share one exact "
                "reviewed-promotion gate."
            )
        final_binding = _strict_behavior_binding(
            self.final_evaluator,
            self.external_behavior_contract,
            name="final_evaluator",
        )
        reload_binding = _strict_behavior_binding(
            self.fresh_reload_evaluator,
            self.external_behavior_contract,
            name="fresh_reload_evaluator",
        )
        if final_binding.manifest_fingerprint != reload_binding.manifest_fingerprint:
            raise Lfm2CpcfCatalogError(
                "V5 final and fresh-reload behavior manifests differ."
            )
        reviewed = self.final_evaluator.evidence
        if (
            reviewed.compiled_model_fold_schema != LFM2_CPCF_V5_MODEL_FOLD_SCHEMA
            or reviewed.full_reviewed_evidence_fingerprint
            != self.expected_full_reviewed_evidence_fingerprint
            or reviewed.parent_cache_key != self.backend.cache.cache_key
            or reviewed.cache_live_parent_fingerprint
            != self.backend.cache.parent_fingerprint
            or reviewed.blank_target_parent_graph_fingerprint
            != self.parent_graph_fingerprint
            or reviewed.trace_adapter_sha256 != trace.evidence_sha256
            or reviewed.bridge_artifact_sha256 != trace.bridge_artifact_sha256
            or reviewed.bridge_profile_sha256 != trace.transport_profile_fingerprint
            or reviewed.target_tokenizer_semantics_sha256
            != trace.target_tokenizer_semantics_sha256
            or reviewed.candidate_tokenizer_fingerprint
            != trace.target_tokenizer_package_fingerprint
            or canonical_sha256(dict(reviewed.compiled_model_fold_policy))
            != canonical_sha256(self.backend.runtime_policy.as_dict())
        ):
            raise Lfm2CpcfCatalogError(
                "V5 cache, parent, trace, bridge, model policy, or reviewed "
                "whole-output authority identities differ."
            )
        table = self.source_tied_table
        torch = _torch()
        if (
            not isinstance(table, torch.Tensor)
            or table.layout != torch.strided
            or table.device.type == "meta"
            or table.ndim != 2
            or table.numel() == 0
            or not torch.is_floating_point(table)
            or table.requires_grad
            or table.grad_fn is not None
            or not bool(torch.isfinite(table.detach()).all().item())
            or int(table.shape[0]) != trace.target_row_capacity
        ):
            raise Lfm2CpcfCatalogError(
                "V5 source_tied_table must be a finite detached dense target-row table."
            )
        object.__setattr__(self, "saver_id", _text(self.saver_id, "saver_id"))
        object.__setattr__(self, "loader_id", _text(self.loader_id, "loader_id"))
        if self.snapshot_config is not None and not isinstance(
            self.snapshot_config,
            DynamicFoldSnapshotConfig,
        ):
            raise Lfm2CpcfCatalogError(
                "snapshot_config must be DynamicFoldSnapshotConfig or None."
            )
        (
            resume_coordinator,
            resume_state,
            resume_admission_sha256,
            resume_contract,
        ) = _v5_resume_selection(
            self.resume_coordinator,
            self.resume_state,
            self.resume_admission_sha256,
        )
        _verify_v5_resume_static_context(
            resume_state,
            backend=self.backend,
            trace=trace,
            source_tied_table=table,
            reviewed=reviewed,
        )
        object.__setattr__(
            self,
            "resume_coordinator",
            resume_coordinator,
        )
        object.__setattr__(self, "resume_state", resume_state)
        object.__setattr__(
            self,
            "resume_admission_sha256",
            resume_admission_sha256,
        )
        publication_forbidden_objects = {
            "selective_backend": self.backend,
            "selective_cache": self.backend.cache,
            "tokenizer_trace_adapter": trace,
            "source_tied_table": table,
            "resume_coordinator": resume_coordinator,
            "resume_state": resume_state,
            "resume_store": (
                None if resume_coordinator is None else resume_coordinator.store
            ),
        }
        callback_sidecars = _resume_sidecar_paths_from_roots(
            (
                ("model_saver", self.model_saver),
                ("model_loader", self.model_loader),
            ),
            forbidden_objects=publication_forbidden_objects,
        )
        if callback_sidecars:
            raise Lfm2CpcfCatalogError(
                "V5 saver/loader callbacks capture a parent-derived, cache, "
                "tokenizer-bridge, or resume resource."
            )
        resource_body = {
            "schema": LFM2_CPCF_V5_RESOURCE_SCHEMA,
            "backend_fingerprint": self.backend.backend_fingerprint,
            "cache_key": self.backend.cache.cache_key,
            "parent_graph_fingerprint": self.parent_graph_fingerprint,
            "trace_adapter_sha256": trace.evidence_sha256,
            "bridge_artifact_sha256": trace.bridge_artifact_sha256,
            "transport_profile_sha256": (trace.transport_profile_fingerprint),
            "source_tied_table_sha256": _tensor_sha256(
                table,
                "source_tied_table",
            ),
            "reviewed_candidate_state_fingerprint": (
                reviewed.candidate_state_fingerprint
            ),
            "reviewed_compiled_model_fold_schema": (
                reviewed.compiled_model_fold_schema
            ),
            "external_behavior_contract_sha256": (
                self.external_behavior_contract.contract_sha256
            ),
            "behavior_gate_manifest_fingerprint": (final_binding.manifest_fingerprint),
            "expected_full_reviewed_evidence_fingerprint": (
                self.expected_full_reviewed_evidence_fingerprint
            ),
            "saver_id": self.saver_id,
            "loader_id": self.loader_id,
            "snapshot_config": (
                None if self.snapshot_config is None else self.snapshot_config.as_dict()
            ),
            "selective_cpu_residency": True,
            "bounded_pinned_host_to_cuda_staging": True,
            "whole_output_reviewed_gate_required": True,
            "fresh_reload_required": True,
            "resume_contract": _plain_json(resume_contract),
            "resume_implementation_sha256": (
                lfm2_cpcf_v5_resume_implementation()["implementation_sha256"]
            ),
        }
        object.__setattr__(
            self,
            "resource_fingerprint",
            canonical_sha256(resource_body),
        )


class Lfm2CpcfSealedEvidence:
    """Executor-safe seal over large parent evidence and target-bound metadata."""

    __slots__ = (
        "_phases",
        "_trace_adapter",
        "_source_tied_table",
        "_blank_target_report",
        "_policy",
        "_cache_key",
        "_phase_sha256",
        "_parent_evidence_sha256",
        "_parent_tensor_seals",
        "_parent_tensor_content_sha256",
        "_parent_tensor_content_aggregate_sha256",
        "_trace_adapter_sha256",
        "_bridge_artifact_sha256",
        "_source_tied_table_sha256",
        "_blank_report_sha256",
        "_policy_sha256",
        "_bundle_sha256",
    )

    def __init__(
        self,
        *,
        phases: Sequence[Lfm2CpcfTargetPhase],
        trace_adapter: Lfm2StrictTokenizerBridgeTraceAdapter,
        source_tied_table: Any,
        blank_target_report: Mapping[str, Any],
        policy: Lfm2CpcfModelFoldPolicy,
    ) -> None:
        ordered = tuple(phases)
        cache_keys = {phase.parent_evidence.cache_key for phase in ordered}
        if len(cache_keys) != 1:
            raise Lfm2CpcfCatalogError(
                "All CPCF phases must share one parent cache key."
            )
        self._phases = ordered
        self._trace_adapter = trace_adapter.verify()
        self._source_tied_table = source_tied_table
        self._blank_target_report = MappingProxyType(dict(blank_target_report))
        self._policy = policy
        self._cache_key = _digest(cache_keys.pop(), "phase cache_key")
        self._phase_sha256 = tuple(
            _phase_metadata(phase)["phase_sha256"] for phase in ordered
        )
        self._parent_evidence_sha256 = tuple(
            _digest(
                phase.parent_evidence.evidence_sha256,
                "phase parent_evidence.evidence_sha256",
            )
            for phase in ordered
        )
        parent_content = _authenticate_parent_tensors(ordered)
        self._parent_tensor_content_sha256 = parent_content
        self._parent_tensor_seals = tuple(
            _parent_tensor_mutation_seal(
                path,
                tensor,
                content_sha256=parent_content[path],
            )
            for path, tensor in _parent_tensor_bindings(ordered)
        )
        self._parent_tensor_content_aggregate_sha256 = canonical_sha256(
            [
                {
                    "path": path,
                    "content_sha256": parent_content[path],
                }
                for path in sorted(parent_content)
            ]
        )
        self._trace_adapter_sha256 = _digest(
            trace_adapter.evidence_sha256,
            "trace_adapter.evidence_sha256",
        )
        self._bridge_artifact_sha256 = _digest(
            trace_adapter.bridge_artifact_sha256,
            "trace_adapter.bridge_artifact_sha256",
        )
        self._source_tied_table_sha256 = _tensor_sha256(
            source_tied_table,
            "source_tied_table",
        )
        self._blank_report_sha256 = _digest(
            blank_target_report.get("report_sha256"),
            "blank_target_report.report_sha256",
        )
        self._policy_sha256 = canonical_sha256(policy.as_dict())
        self._bundle_sha256 = canonical_sha256(self._body())
        self.verify()

    @property
    def cache_key(self) -> str:
        return self._cache_key

    @property
    def bundle_sha256(self) -> str:
        return self._bundle_sha256

    @property
    def phases(self) -> tuple[Lfm2CpcfTargetPhase, ...]:
        return self._phases

    @property
    def trace_adapter(self) -> Lfm2StrictTokenizerBridgeTraceAdapter:
        return self._trace_adapter

    @property
    def source_tied_table(self) -> Any:
        return self._source_tied_table

    @property
    def blank_target_report(self) -> Mapping[str, Any]:
        return self._blank_target_report

    @property
    def policy(self) -> Lfm2CpcfModelFoldPolicy:
        return self._policy

    def _body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CPCF_CATALOG_SCHEMA,
            "cache_key": self._cache_key,
            "ordered_phase_sha256": list(self._phase_sha256),
            "ordered_parent_evidence_sha256": list(self._parent_evidence_sha256),
            "parent_tensor_content_aggregate_sha256": (
                self._parent_tensor_content_aggregate_sha256
            ),
            "parent_tensor_seal_schema": (LFM2_CPCF_PARENT_TENSOR_SEAL_SCHEMA),
            "trace_adapter_sha256": self._trace_adapter_sha256,
            "bridge_artifact_sha256": self._bridge_artifact_sha256,
            "source_tied_table_sha256": self._source_tied_table_sha256,
            "blank_target_report_sha256": self._blank_report_sha256,
            "policy_sha256": self._policy_sha256,
            "parent_phase_tensor_rehash_required": True,
            "parent_tensor_storage_object_version_sealed": True,
        }

    def verify(self) -> Lfm2CpcfSealedEvidence:
        self._trace_adapter.verify()
        if (
            self._trace_adapter.evidence_sha256 != self._trace_adapter_sha256
            or self._trace_adapter.bridge_artifact_sha256
            != self._bridge_artifact_sha256
        ):
            raise Lfm2CpcfCatalogError(
                "Tokenizer bridge trace-adapter identity changed."
            )
        identities = tuple(phase.identity for phase in self._phases)
        if len(set(identities)) != len(identities):
            raise Lfm2CpcfCatalogError("CPCF target phase identities repeat.")
        observed_ordinals = {identity[0] for identity in identities}
        expected_ordinals = {
            int(record.ordinal) for record in self._trace_adapter.records
        }
        if observed_ordinals != expected_ordinals:
            raise Lfm2CpcfCatalogError(
                "CPCF phases do not cover every tokenizer-adapter record."
            )
        for phase in self._phases:
            record = self._trace_adapter.records[phase.batch_index]
            if (
                phase.parent_evidence.cache_key != self._cache_key
                or phase.donor_input_sha256 != record.donor_cache_input_sha256
                or phase.target_alignment_sha256 != record.alignment_sha256
                or phase.record_id != record.record_id
                or phase.split != record.split
            ):
                raise Lfm2CpcfCatalogError(
                    "CPCF phase differs from its strict tokenizer record."
                )
        observed_phase_sha = tuple(
            _phase_metadata(phase)["phase_sha256"] for phase in self._phases
        )
        if observed_phase_sha != self._phase_sha256:
            raise Lfm2CpcfCatalogError("CPCF ordered phase seal changed.")
        if (
            tuple(phase.parent_evidence.evidence_sha256 for phase in self._phases)
            != self._parent_evidence_sha256
        ):
            raise Lfm2CpcfCatalogError("CPCF parent evidence identity changed.")
        self._verify_parent_tensor_mutation_seals()
        if (
            _tensor_sha256(
                self._source_tied_table,
                "source_tied_table",
            )
            != self._source_tied_table_sha256
        ):
            raise Lfm2CpcfCatalogError("CPCF source tied table changed.")
        blank = dict(self._blank_target_report)
        blank_sha = blank.pop("report_sha256", None)
        if (
            blank_sha != self._blank_report_sha256
            or canonical_sha256(blank) != self._blank_report_sha256
        ):
            raise Lfm2CpcfCatalogError("CPCF blank-target seal changed.")
        if canonical_sha256(self._policy.as_dict()) != self._policy_sha256:
            raise Lfm2CpcfCatalogError("CPCF model-fold policy changed.")
        if canonical_sha256(self._body()) != self._bundle_sha256:
            raise Lfm2CpcfCatalogError("CPCF evidence bundle seal changed.")
        return self

    def _verify_parent_tensor_mutation_seals(self) -> None:
        current = dict(_parent_tensor_bindings(self._phases))
        if set(current) != {seal.path for seal in self._parent_tensor_seals}:
            raise Lfm2CpcfCatalogError("CPCF parent evidence tensor paths changed.")
        for seal in self._parent_tensor_seals:
            observed = _parent_tensor_mutation_seal(
                seal.path,
                current[seal.path],
                content_sha256=seal.content_sha256,
            )
            if observed != seal:
                raise Lfm2CpcfCatalogError(
                    "CPCF parent evidence tensor storage/object/version seal "
                    f"changed at {seal.path!r}."
                )

    def verify_parent_tensor_integrity(
        self,
    ) -> Lfm2CpcfSealedEvidence:
        """Rehash all parent evidence at the last pre-compile boundary."""

        self.verify()
        observed = _authenticate_parent_tensors(self._phases)
        observed_aggregate = canonical_sha256(
            [
                {
                    "path": path,
                    "content_sha256": observed[path],
                }
                for path in sorted(observed)
            ]
        )
        if (
            dict(observed) != dict(self._parent_tensor_content_sha256)
            or observed_aggregate != self._parent_tensor_content_aggregate_sha256
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF parent evidence failed authenticated tensor integrity "
                "revalidation."
            )
        # Catch ordinary in-place writes that race with the canonical scan.
        self._verify_parent_tensor_mutation_seals()
        return self


class Lfm2CpcfV5SelectiveSealedEvidence:
    """Small executor seal over a disk-backed selective v5 evidence root."""

    __slots__ = (
        "_backend",
        "_trace_adapter",
        "_source_tied_table",
        "_blank_target_report",
        "_reviewed_candidate_state_fingerprint",
        "_resume_coordinator",
        "_resume_state",
        "_resume_admission_sha256",
        "_resume_contract",
        "_resource_fingerprint",
        "_selected_target_sha256",
        "_trace_adapter_sha256",
        "_source_tied_table_sha256",
        "_blank_report_sha256",
        "_bundle_sha256",
    )

    def __init__(
        self,
        *,
        backend: Lfm2CpcfV5SelectiveBackend,
        trace_adapter: Lfm2StrictTokenizerBridgeTraceAdapter,
        source_tied_table: Any,
        blank_target_report: Mapping[str, Any],
        reviewed_candidate_state_fingerprint: str,
        resume_coordinator: Lfm2CpcfV5ResumeCoordinator | None = None,
        resume_state: Lfm2CpcfV5ResumeState | None = None,
        resume_admission_sha256: str | None = None,
        resource_fingerprint: str | None = None,
        selected_target_sha256: str | None = None,
    ) -> None:
        if type(backend) is not Lfm2CpcfV5SelectiveBackend:
            raise Lfm2CpcfCatalogError("V5 evidence backend has the wrong type.")
        self._backend = backend.verify()
        self._trace_adapter = trace_adapter.verify()
        self._source_tied_table = source_tied_table
        self._blank_target_report = MappingProxyType(
            deepcopy(dict(blank_target_report))
        )
        self._reviewed_candidate_state_fingerprint = _digest(
            reviewed_candidate_state_fingerprint,
            "reviewed_candidate_state_fingerprint",
        )
        (
            self._resume_coordinator,
            self._resume_state,
            self._resume_admission_sha256,
            self._resume_contract,
        ) = _v5_resume_selection(
            resume_coordinator,
            resume_state,
            resume_admission_sha256,
        )
        self._resource_fingerprint = (
            None
            if resource_fingerprint is None
            else _digest(resource_fingerprint, "resource_fingerprint")
        )
        self._selected_target_sha256 = (
            None
            if selected_target_sha256 is None
            else _digest(selected_target_sha256, "selected_target_sha256")
        )
        if self._resume_coordinator is not None and (
            self._resource_fingerprint is None or self._selected_target_sha256 is None
        ):
            raise Lfm2CpcfCatalogError(
                "V5 checkpoint publication or continuation requires exact "
                "resource and selected-target bindings."
            )
        self._trace_adapter_sha256 = _digest(
            self._trace_adapter.evidence_sha256,
            "trace_adapter.evidence_sha256",
        )
        self._source_tied_table_sha256 = _tensor_sha256(
            source_tied_table,
            "source_tied_table",
        )
        self._blank_report_sha256 = _digest(
            blank_target_report.get("report_sha256"),
            "blank_target_report.report_sha256",
        )
        self._bundle_sha256 = canonical_sha256(self._body())
        self.verify()

    @property
    def backend(self) -> Lfm2CpcfV5SelectiveBackend:
        return self._backend

    @property
    def cache_key(self) -> str:
        return self._backend.cache.cache_key

    @property
    def bundle_sha256(self) -> str:
        return self._bundle_sha256

    @property
    def trace_adapter(self) -> Lfm2StrictTokenizerBridgeTraceAdapter:
        return self._trace_adapter

    @property
    def source_tied_table(self) -> Any:
        return self._source_tied_table

    @property
    def blank_target_report(self) -> Mapping[str, Any]:
        return self._blank_target_report

    @property
    def reviewed_candidate_state_fingerprint(self) -> str:
        return self._reviewed_candidate_state_fingerprint

    @property
    def resume_coordinator(self) -> Lfm2CpcfV5ResumeCoordinator | None:
        return self._resume_coordinator

    @property
    def resume_state(self) -> Lfm2CpcfV5ResumeState | None:
        return self._resume_state

    @property
    def resume_admission_sha256(self) -> str | None:
        return self._resume_admission_sha256

    @property
    def resume_contract(self) -> Mapping[str, Any]:
        return self._resume_contract

    @property
    def resource_fingerprint(self) -> str | None:
        return self._resource_fingerprint

    @property
    def selected_target_sha256(self) -> str | None:
        return self._selected_target_sha256

    def _body(self) -> dict[str, Any]:
        return {
            "schema": LFM2_CPCF_V5_EVIDENCE_SCHEMA,
            "backend_fingerprint": self._backend.backend_fingerprint,
            "cache_key": self.cache_key,
            "trace_adapter_sha256": self._trace_adapter_sha256,
            "bridge_artifact_sha256": (self._trace_adapter.bridge_artifact_sha256),
            "source_tied_table_sha256": (self._source_tied_table_sha256),
            "blank_target_report_sha256": self._blank_report_sha256,
            "reviewed_candidate_state_fingerprint": (
                self._reviewed_candidate_state_fingerprint
            ),
            "resume_contract": _plain_json(self._resume_contract),
            "resume_implementation_sha256": (
                lfm2_cpcf_v5_resume_implementation()["implementation_sha256"]
            ),
            "resource_fingerprint": self._resource_fingerprint,
            "selected_target_sha256": self._selected_target_sha256,
            "parent_phase_tensors_materialized_in_stack": False,
            "selective_provider_opened_only_for_compile": True,
            "resume_parent_or_cache_tensor_sidecars": 0,
        }

    def verify(self) -> Lfm2CpcfV5SelectiveSealedEvidence:
        self._backend.verify()
        self._trace_adapter.verify()
        (
            coordinator,
            state,
            admission_sha256,
            resume_contract,
        ) = _v5_resume_selection(
            self._resume_coordinator,
            self._resume_state,
            self._resume_admission_sha256,
        )
        if (
            self._trace_adapter.evidence_sha256 != self._trace_adapter_sha256
            or _tensor_sha256(
                self._source_tied_table,
                "source_tied_table",
            )
            != self._source_tied_table_sha256
            or coordinator is not self._resume_coordinator
            or (
                state is not None
                and (
                    self._resume_state is None
                    or state.frontier_reference != self._resume_state.frontier_reference
                    or state.checkpoint_reference
                    != self._resume_state.checkpoint_reference
                    or state.frontier.frontier_sha256
                    != self._resume_state.frontier.frontier_sha256
                )
            )
            or admission_sha256 != self._resume_admission_sha256
            or _plain_json(resume_contract) != _plain_json(self._resume_contract)
        ):
            raise Lfm2CpcfCatalogError(
                "V5 trace adapter, tied source table, or admitted resume "
                "selection changed."
            )
        blank = dict(self._blank_target_report)
        report_sha256 = blank.pop("report_sha256", None)
        if (
            report_sha256 != self._blank_report_sha256
            or canonical_sha256(blank) != self._blank_report_sha256
            or canonical_sha256(self._body()) != self._bundle_sha256
        ):
            raise Lfm2CpcfCatalogError(
                "V5 blank-target or evidence bundle seal changed."
            )
        return self

    def verify_parent_tensor_integrity(
        self,
    ) -> Lfm2CpcfV5SelectiveSealedEvidence:
        """V5 has no resident parent phase tree; rehash its small roots."""

        return self.verify()

    def open_provider(self) -> Lfm2CpcfSelectiveEvidenceProvider:
        self.verify()
        return self._backend.open_provider()


class Lfm2CpcfEvidenceProvider:
    """Provide sealed CPCF evidence without regenerating parent traces."""

    def __init__(
        self,
        evidence: Lfm2CpcfSealedEvidence | Lfm2CpcfV5SelectiveSealedEvidence,
    ) -> None:
        if not isinstance(
            evidence,
            (Lfm2CpcfSealedEvidence, Lfm2CpcfV5SelectiveSealedEvidence),
        ):
            raise Lfm2CpcfCatalogError(
                "evidence must be sealed legacy or selective-v5 CPCF evidence."
            )
        self.evidence = evidence.verify()
        self.prepare_count = 0
        self.refresh_count = 0
        self.restore_count = 0

    @property
    def base_fingerprint(self) -> str:
        return self.evidence.cache_key

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
    ) -> FoldEvidence:
        del context
        if action.method_id != DENSE_BUDGET_MODEL_METHOD:
            raise Lfm2CpcfCatalogError(
                "CPCF evidence received an unsupported fold method."
            )
        self.evidence.verify()
        self.prepare_count += 1
        return FoldEvidence(
            self.evidence.bundle_sha256,
            payload=self.evidence,
            channels=action.evidence_requirements,
            parent_forward_count=0,
            child_forward_count=0,
            root_fingerprint=self.evidence.cache_key,
        )

    def refresh(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        transaction_report: Mapping[str, Any],
    ) -> None:
        del action, context, transaction_report
        self.evidence.verify()
        self.refresh_count += 1

    def restore(
        self,
        context: FoldExecutionContext,
        state_fingerprint: str,
        action_ids: Sequence[str],
    ) -> None:
        del action_ids
        if context.child_state_fingerprint != _digest(
            state_fingerprint,
            "state_fingerprint",
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF evidence restore observed an unrestored child."
            )
        self.evidence.verify()
        self.restore_count += 1


@dataclass(frozen=True, slots=True)
class _CpcfModelSnapshot:
    parameter_bindings: tuple[tuple[str, Any, bool], ...]
    parameter_storage_identity: Mapping[str, tuple[Any, ...]]
    buffer_bindings: tuple[tuple[str, Any], ...]
    buffer_storage_identity: Mapping[str, tuple[Any, ...]]
    tensor_values: Mapping[int, Any]
    module_modes: tuple[tuple[Any, bool], ...]
    state_fingerprint: str
    runtime_fingerprint: str
    portable_topology_fingerprint: str


def _named_parameters(model: Any) -> tuple[tuple[str, Any], ...]:
    try:
        return tuple(
            model.named_parameters(
                recurse=True,
                remove_duplicate=False,
            )
        )
    except TypeError:  # pragma: no cover - old Torch compatibility
        return tuple(model.named_parameters(recurse=True))


def _named_buffers(model: Any) -> tuple[tuple[str, Any], ...]:
    try:
        return tuple(
            model.named_buffers(
                recurse=True,
                remove_duplicate=False,
            )
        )
    except TypeError:  # pragma: no cover - old Torch compatibility
        return tuple(model.named_buffers(recurse=True))


def _tensor_storage_identity(tensor: Any) -> tuple[Any, ...]:
    storage = tensor.untyped_storage()
    return (
        int(getattr(storage, "_cdata", 0)),
        int(storage.data_ptr()),
        int(storage.nbytes()),
        int(tensor.storage_offset()),
        int(tensor.numel()),
        int(tensor.element_size()),
        tuple(int(size) for size in tensor.shape),
        tuple(int(value) for value in tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
    )


def _parameter_storage_identity(parameter: Any) -> tuple[Any, ...]:
    return _tensor_storage_identity(parameter)


def _portable_model_topology_fingerprint(model: Any) -> str:
    parameters = _named_parameters(model)
    buffers = _named_buffers(model)
    object_ordinals: dict[int, int] = {}
    storage_ordinals: dict[tuple[Any, ...], int] = {}
    rows: list[dict[str, Any]] = []
    for kind, values in (("parameter", parameters), ("buffer", buffers)):
        for name, tensor in values:
            object_ordinal = object_ordinals.setdefault(
                id(tensor),
                len(object_ordinals),
            )
            storage = tensor.untyped_storage()
            storage_key = (
                str(tensor.device),
                int(getattr(storage, "_cdata", 0)),
            )
            storage_ordinal = storage_ordinals.setdefault(
                storage_key,
                len(storage_ordinals),
            )
            rows.append(
                {
                    "kind": kind,
                    "name": name,
                    "object_alias_ordinal": object_ordinal,
                    "storage_alias_ordinal": storage_ordinal,
                    "storage_offset": int(tensor.storage_offset()),
                    "shape": [int(size) for size in tensor.shape],
                    "stride": [int(value) for value in tensor.stride()],
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "requires_grad": bool(tensor.requires_grad),
                }
            )
    module_rows = [
        {
            "type": (f"{type(module).__module__}.{type(module).__qualname__}"),
            "training": bool(module.training),
        }
        for module in model.modules()
    ]
    return canonical_sha256(
        {
            "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
            "tensors": rows,
            "modules": module_rows,
        }
    )


@dataclass(frozen=True, slots=True)
class _CompileProtectedTensorSeal:
    path: str
    tensor: Any
    object_id: int
    storage_identity: tuple[Any, ...]
    version: int


def _compile_protected_tensor_bindings(
    *,
    phases: Sequence[Lfm2CpcfTargetPhase],
    source_tied_table: Any,
    parent_model: Any,
    live_child_model: Any,
    staging_child_model: Any,
) -> tuple[tuple[str, Any], ...]:
    bindings: list[tuple[str, Any]] = list(_parent_tensor_bindings(phases))
    for phase in phases:
        phase_prefix = f"phase[{phase.batch_index}:{phase.phase}]"
        bindings.extend(
            (
                (f"{phase_prefix}.input_ids", phase.input_ids),
                (f"{phase_prefix}.attention_mask", phase.attention_mask),
                (f"{phase_prefix}.position_ids", phase.position_ids),
            )
        )
    bindings.append(("source_tied_table", source_tied_table))
    bindings.extend(
        (f"parent.parameter.{name}", parameter)
        for name, parameter in _named_parameters(parent_model)
    )
    bindings.extend(
        (f"parent.buffer.{name}", buffer)
        for name, buffer in _named_buffers(parent_model)
    )
    # The compiler receives only the staging child.  Sealing the live blank as
    # well turns accidental closure/global access into a fail-closed event.
    bindings.extend(
        (f"live_child.parameter.{name}", parameter)
        for name, parameter in _named_parameters(live_child_model)
    )
    bindings.extend(
        (f"live_child.buffer.{name}", buffer)
        for name, buffer in _named_buffers(live_child_model)
    )
    bindings.extend(
        (f"staging_child.buffer.{name}", buffer)
        for name, buffer in _named_buffers(staging_child_model)
    )
    paths = tuple(path for path, _tensor in bindings)
    if len(set(paths)) != len(paths):
        raise Lfm2CpcfCatalogError("CPCF compile write-guard paths are not unique.")
    return tuple(bindings)


def _iter_tensors(value: Any) -> tuple[Any, ...]:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, Mapping):
        tensors: list[Any] = []
        for item in value.values():
            tensors.extend(_iter_tensors(item))
        return tuple(tensors)
    if isinstance(value, (tuple, list)):
        tensors = []
        for item in value:
            tensors.extend(_iter_tensors(item))
        return tuple(tensors)
    return ()


class _CompileSealedStorageWriteGuard:
    """Torch-dispatch firewall over every read-only compiler storage."""

    def __init__(
        self,
        bindings: Sequence[tuple[str, Any]],
        *,
        writable_bindings: Sequence[tuple[str, Any]],
    ) -> None:
        torch = _torch()
        protected_seals: list[_CompileProtectedTensorSeal] = []
        writable_seals: list[_CompileProtectedTensorSeal] = []
        for items, destination in (
            (tuple(bindings), protected_seals),
            (tuple(writable_bindings), writable_seals),
        ):
            for path, tensor in items:
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.layout != torch.strided
                    or tensor.device.type == "meta"
                ):
                    raise Lfm2CpcfCatalogError(
                        f"CPCF compile-protected tensor {path!r} is not "
                        "materialized dense storage."
                    )
                version = getattr(tensor, "_version", None)
                if isinstance(version, bool) or not isinstance(version, int):
                    raise Lfm2CpcfCatalogError(
                        f"CPCF compile-protected tensor {path!r} has no "
                        "mutation version."
                    )
                destination.append(
                    _CompileProtectedTensorSeal(
                        path=path,
                        tensor=tensor,
                        object_id=id(tensor),
                        storage_identity=_tensor_storage_identity(tensor),
                        version=version,
                    )
                )
        if len({seal.path for seal in protected_seals}) != len(protected_seals) or len(
            {seal.path for seal in writable_seals}
        ) != len(writable_seals):
            raise Lfm2CpcfCatalogError(
                "CPCF compile guard tensor paths are not unique."
            )
        protected_storage_roots = {
            (seal.storage_identity[0], seal.storage_identity[-1])
            for seal in protected_seals
        }
        writable_storage_roots = {
            (seal.storage_identity[0], seal.storage_identity[-1])
            for seal in writable_seals
        }
        if not writable_seals or protected_storage_roots.intersection(
            writable_storage_roots
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF staging writable storage is empty or aliases sealed "
                "read-only compiler storage."
            )
        self._seals = tuple(protected_seals)
        self._writable_seals = tuple(writable_seals)
        self._violations: list[dict[str, Any]] = []
        self._writable_events: list[dict[str, Any]] = []
        try:
            from torch.utils._python_dispatch import TorchDispatchMode
        except ImportError as exc:  # pragma: no cover - version boundary
            raise Lfm2CpcfCatalogError(
                "PyTorch TorchDispatchMode is required for CPCF compile "
                "write protection."
            ) from exc
        guard = self

        class _Mode(TorchDispatchMode):
            def __torch_dispatch__(
                self,
                func: Any,
                types: Any,
                args: tuple[Any, ...] = (),
                kwargs: Mapping[str, Any] | None = None,
            ) -> Any:
                del types
                resolved_kwargs = dict(kwargs or {})
                written = guard._written_tensors(
                    func,
                    args,
                    resolved_kwargs,
                )
                protected = guard._protected_paths(written)
                if protected:
                    operation = str(func)
                    guard._violations.append(
                        {
                            "operation": operation,
                            "protected_paths": sorted(protected),
                        }
                    )
                    raise Lfm2CpcfCatalogError(
                        "CPCF compiler attempted a write to sealed storage "
                        f"via {operation}: {sorted(protected)!r}."
                    )
                writable = guard._protected_paths(
                    written,
                    seals=guard._writable_seals,
                )
                result = func(*args, **resolved_kwargs)
                if writable:
                    guard._writable_events.append(
                        {
                            "operation": str(func),
                            "writable_paths": sorted(writable),
                        }
                    )
                return result

        self.mode = _Mode()

    @staticmethod
    def _written_tensors(
        func: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        schema = getattr(func, "_schema", None)
        arguments = getattr(schema, "arguments", ())
        positional_index = 0
        written: list[Any] = []
        for argument in arguments:
            value: Any
            if not bool(argument.kwarg_only) and positional_index < len(args):
                value = args[positional_index]
                positional_index += 1
            elif argument.name in kwargs:
                value = kwargs[argument.name]
            else:
                if not bool(argument.kwarg_only):
                    positional_index += 1
                continue
            alias_info = getattr(argument, "alias_info", None)
            if alias_info is not None and bool(alias_info.is_write):
                written.extend(_iter_tensors(value))
        return tuple(written)

    def _protected_paths(
        self,
        tensors: Sequence[Any],
        *,
        seals: Sequence[_CompileProtectedTensorSeal] | None = None,
    ) -> set[str]:
        protected: set[str] = set()
        for tensor in tensors:
            identity = _tensor_storage_identity(tensor)
            storage_cdata = identity[0]
            data_ptr = identity[1]
            nbytes = identity[2]
            device = identity[-1]
            for seal in self._seals if seals is None else seals:
                sealed_identity = seal.storage_identity
                sealed_cdata = sealed_identity[0]
                sealed_ptr = sealed_identity[1]
                sealed_nbytes = sealed_identity[2]
                sealed_device = sealed_identity[-1]
                same_storage = (
                    storage_cdata != 0
                    and storage_cdata == sealed_cdata
                    and device == sealed_device
                )
                overlapping_range = (
                    nbytes > 0
                    and sealed_nbytes > 0
                    and device == sealed_device
                    and data_ptr < sealed_ptr + sealed_nbytes
                    and sealed_ptr < data_ptr + nbytes
                )
                if same_storage or overlapping_range:
                    protected.add(seal.path)
        return protected

    def assert_clean(
        self,
        current_bindings: Sequence[tuple[str, Any]],
        current_writable_bindings: Sequence[tuple[str, Any]],
        *,
        require_complete_writes: bool,
    ) -> Mapping[str, Any]:
        if self._violations:
            first = self._violations[0]
            raise Lfm2CpcfCatalogError(
                "CPCF compiler attempted a sealed-storage write even if "
                "the compiler swallowed the dispatch error: "
                f"{first['operation']} -> {first['protected_paths']!r}."
            )
        current_items = tuple(current_bindings)
        current = dict(current_items)
        writable_items = tuple(current_writable_bindings)
        current_writable = dict(writable_items)
        if len(current) != len(current_items) or set(current) != {
            seal.path for seal in self._seals
        }:
            raise Lfm2CpcfCatalogError(
                "CPCF compile-protected tensor bindings changed."
            )
        if len(current_writable) != len(writable_items) or set(current_writable) != {
            seal.path for seal in self._writable_seals
        }:
            raise Lfm2CpcfCatalogError("CPCF staging writable tensor bindings changed.")
        for seal in self._seals:
            tensor = current[seal.path]
            version = getattr(tensor, "_version", None)
            if (
                tensor is not seal.tensor
                or id(tensor) != seal.object_id
                or _tensor_storage_identity(tensor) != seal.storage_identity
                or version != seal.version
            ):
                raise Lfm2CpcfCatalogError(
                    "CPCF compile-protected tensor object, storage, or "
                    f"version changed at {seal.path!r}."
                )
        for seal in self._writable_seals:
            tensor = current_writable[seal.path]
            if (
                tensor is not seal.tensor
                or id(tensor) != seal.object_id
                or _tensor_storage_identity(tensor) != seal.storage_identity
            ):
                raise Lfm2CpcfCatalogError(
                    "CPCF staging writable tensor object or storage changed "
                    f"at {seal.path!r}."
                )
        written_paths = {
            path for event in self._writable_events for path in event["writable_paths"]
        }
        missing_storage_paths = []
        writable_groups: dict[tuple[Any, ...], list[str]] = {}
        for seal in self._writable_seals:
            storage_root = (
                seal.storage_identity[0],
                seal.storage_identity[-1],
            )
            writable_groups.setdefault(storage_root, []).append(seal.path)
        for paths in writable_groups.values():
            if not set(paths).intersection(written_paths):
                missing_storage_paths.append(sorted(paths))
        if missing_storage_paths and require_complete_writes:
            raise Lfm2CpcfCatalogError(
                "CPCF compiler did not write every unique staging target "
                f"parameter storage: {missing_storage_paths!r}."
            )
        ordered_events = sorted(
            (
                {
                    "operation": str(event["operation"]),
                    "writable_paths": list(event["writable_paths"]),
                }
                for event in self._writable_events
            ),
            key=lambda event: (
                event["operation"],
                tuple(event["writable_paths"]),
            ),
        )
        body = {
            "schema": LFM2_CPCF_COMPILE_WRITE_GUARD_SCHEMA,
            "protected_binding_count": len(self._seals),
            "protected_unique_storage_count": len(
                {seal.storage_identity[:3] for seal in self._seals}
            ),
            "attempted_write_count": 0,
            "all_protected_bindings_unchanged": True,
            "compiler_write_attempts_rejected": True,
            "writable_parameter_path_count": len(self._writable_seals),
            "writable_unique_storage_count": len(writable_groups),
            "authenticated_writable_parameter_paths": sorted(written_paths),
            "authenticated_writable_operation_count": len(self._writable_events),
            "authenticated_writable_event_aggregate_sha256": (
                canonical_sha256(ordered_events)
            ),
            "all_unique_writable_parameter_storages_written": (
                not missing_storage_paths
            ),
            "writable_parameter_objects_and_storages_preserved": True,
        }
        return MappingProxyType({**body, "report_sha256": canonical_sha256(body)})


def _capture_model(model: Any) -> _CpcfModelSnapshot:
    parameters = _named_parameters(model)
    buffers = _named_buffers(model)
    if any(parameter.grad is not None for _name, parameter in parameters):
        raise Lfm2CpcfCatalogError(
            "CPCF requires a zero-gradient target before folding."
        )
    values: dict[int, Any] = {}
    for _name, tensor in (*parameters, *buffers):
        values.setdefault(
            id(tensor),
            tensor.detach().to(device="cpu").contiguous().clone(),
        )
    return _CpcfModelSnapshot(
        parameter_bindings=tuple(
            (name, parameter, bool(parameter.requires_grad))
            for name, parameter in parameters
        ),
        parameter_storage_identity=MappingProxyType(
            {
                name: _parameter_storage_identity(parameter)
                for name, parameter in parameters
            }
        ),
        buffer_bindings=buffers,
        buffer_storage_identity=MappingProxyType(
            {name: _tensor_storage_identity(buffer) for name, buffer in buffers}
        ),
        tensor_values=MappingProxyType(values),
        module_modes=tuple(
            (module, bool(module.training)) for module in model.modules()
        ),
        state_fingerprint=fingerprint_model_state(model),
        runtime_fingerprint=fingerprint_model_runtime_state(model),
        portable_topology_fingerprint=(_portable_model_topology_fingerprint(model)),
    )


def _restore_model(model: Any, snapshot: _CpcfModelSnapshot) -> None:
    torch = _torch()
    parameters = _named_parameters(model)
    buffers = _named_buffers(model)
    if tuple((name, id(value)) for name, value in parameters) != tuple(
        (name, id(value)) for name, value, _requires_grad in snapshot.parameter_bindings
    ) or tuple((name, id(value)) for name, value in buffers) != tuple(
        (name, id(value)) for name, value in snapshot.buffer_bindings
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF target tensor topology changed before rollback."
        )
    if {
        name: _parameter_storage_identity(parameter) for name, parameter in parameters
    } != dict(snapshot.parameter_storage_identity):
        raise Lfm2CpcfCatalogError(
            "CPCF target parameter storage changed before rollback."
        )
    if {name: _tensor_storage_identity(buffer) for name, buffer in buffers} != dict(
        snapshot.buffer_storage_identity
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF target buffer storage changed before rollback."
        )
    with torch.no_grad():
        restored: set[int] = set()
        for _name, tensor in (*parameters, *buffers):
            if id(tensor) in restored:
                continue
            source = snapshot.tensor_values[id(tensor)]
            if (
                tuple(tensor.shape) != tuple(source.shape)
                or tensor.dtype != source.dtype
            ):
                raise Lfm2CpcfCatalogError(
                    "CPCF target tensor shape or dtype changed before rollback."
                )
            tensor.copy_(source.to(device=tensor.device, dtype=tensor.dtype))
            restored.add(id(tensor))
        for _name, parameter, requires_grad in snapshot.parameter_bindings:
            parameter.requires_grad_(requires_grad)
            parameter.grad = None
    for module, training in snapshot.module_modes:
        module.training = training
    if (
        fingerprint_model_state(model) != snapshot.state_fingerprint
        or fingerprint_model_runtime_state(model) != snapshot.runtime_fingerprint
        or _portable_model_topology_fingerprint(model)
        != snapshot.portable_topology_fingerprint
    ):
        raise Lfm2CpcfCatalogError("CPCF rollback did not restore the exact target.")


def _assert_model_matches_snapshot_topology(
    model: Any,
    snapshot: _CpcfModelSnapshot,
    *,
    name: str,
) -> None:
    parameters = _named_parameters(model)
    buffers = _named_buffers(model)
    if tuple((path, id(value)) for path, value in parameters) != tuple(
        (path, id(value)) for path, value, _requires_grad in snapshot.parameter_bindings
    ):
        raise Lfm2CpcfCatalogError(f"CPCF {name} Parameter objects or paths changed.")
    if tuple((path, id(value)) for path, value in buffers) != tuple(
        (path, id(value)) for path, value in snapshot.buffer_bindings
    ):
        raise Lfm2CpcfCatalogError(f"CPCF {name} buffer objects or paths changed.")
    if {
        path: _parameter_storage_identity(parameter) for path, parameter in parameters
    } != dict(snapshot.parameter_storage_identity):
        raise Lfm2CpcfCatalogError(f"CPCF {name} Parameter storage topology changed.")
    if {path: _tensor_storage_identity(buffer) for path, buffer in buffers} != dict(
        snapshot.buffer_storage_identity
    ):
        raise Lfm2CpcfCatalogError(f"CPCF {name} buffer storage topology changed.")
    if (
        fingerprint_model_runtime_state(model) != snapshot.runtime_fingerprint
        or _portable_model_topology_fingerprint(model)
        != snapshot.portable_topology_fingerprint
    ):
        raise Lfm2CpcfCatalogError(f"CPCF {name} runtime topology changed.")


def _model_storage_roots(model: Any) -> set[tuple[str, int]]:
    roots: set[tuple[str, int]] = set()
    for _name, tensor in (
        *_named_parameters(model),
        *_named_buffers(model),
    ):
        storage = tensor.untyped_storage()
        roots.add(
            (
                str(tensor.device),
                int(getattr(storage, "_cdata", 0)),
            )
        )
    return roots


def _make_disposable_staging_blank(
    live_model: Any,
    live_snapshot: _CpcfModelSnapshot,
) -> tuple[Any, _CpcfModelSnapshot]:
    try:
        staging_model = deepcopy(live_model)
    except BaseException as exc:
        raise Lfm2CpcfCatalogError(
            "CPCF could not create a disposable staging blank."
        ) from exc
    staging_snapshot = _capture_model(staging_model)
    if (
        staging_snapshot.state_fingerprint != live_snapshot.state_fingerprint
        or staging_snapshot.portable_topology_fingerprint
        != live_snapshot.portable_topology_fingerprint
        or _model_storage_roots(staging_model).intersection(
            _model_storage_roots(live_model)
        )
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF staging blank is not an exact storage-disjoint copy of "
            "the authenticated live blank."
        )
    return staging_model, staging_snapshot


def _verify_v5_resume_live_context(
    *,
    sealed: Lfm2CpcfV5SelectiveSealedEvidence,
    parent_model: Any,
    live_model: Any,
    selected_target: Lfm2DenseBudgetedSelectedTarget,
) -> None:
    """Bind an admitted prefix to the exact live parent and fresh blank."""

    if sealed.resume_state is None:
        return
    state = sealed.resume_state
    source = _lfm2_cpcf_model_fold_module._snapshot_model_auth(parent_model)
    target = _lfm2_cpcf_model_fold_module._snapshot_model(live_model)
    target_topology = build_lfm2_cpcf_target_topology_snapshot(
        prefix_model=target.tensor_values,
        layer_count=sealed.backend.expected_layer_count,
        snapshot_metadata=(
            _lfm2_cpcf_model_fold_module._snapshot_metadata_plain(target)
        ),
    )
    context = state.frontier.context
    expected = {
        "source_model_state_sha256": source.state_fingerprint,
        "source_model_inventory_sha256": source.full_inventory_sha256,
        "cache_sha256": sealed.cache_key,
        "selective_catalog_sha256": (sealed.backend.expected_selective_catalog_sha256),
        "selective_source_sha256": (sealed.backend.expected_selective_source_sha256),
        "selective_provider_sha256": (
            sealed.backend.expected_selective_provider_sha256
        ),
        "trace_sha256": sealed.trace_adapter.evidence_sha256,
        "tokenizer_sha256": (sealed.trace_adapter.target_tokenizer_package_fingerprint),
        "token_bridge_sha256": (sealed.trace_adapter.bridge_artifact_sha256),
        "source_tied_table_sha256": _tensor_sha256(
            sealed.source_tied_table,
            "source_tied_table",
        ),
        "blank_target_report_sha256": (sealed.blank_target_report["report_sha256"]),
        "blank_target_state_sha256": target.state_fingerprint,
        "target_topology_sha256": target_topology["topology_sha256"],
        "planned_target_unique_parameters": (
            selected_target.exact_target_unique_parameters
        ),
        "runtime_policy_sha256": canonical_sha256(
            sealed.backend.runtime_policy.as_dict()
        ),
        "physical_rss_policy_sha256": (
            build_lfm2_cpcf_v5_physical_rss_policy_sha256(
                full_phase_tree_logical_bytes=(
                    sealed.backend.runtime_policy.full_phase_tree_logical_bytes
                ),
                minimum_reduction_fraction=(
                    sealed.backend.runtime_policy.minimum_physical_rss_reduction_fraction
                ),
                maximum_increase_bytes=(
                    sealed.backend.runtime_policy.maximum_physical_rss_increase_bytes
                ),
                maximum_process_rss_bytes=(
                    sealed.backend.runtime_policy.maximum_process_rss_bytes
                ),
            )
        ),
        "compiler_implementation_sha256": (
            lfm2_cpcf_v5_compiler_implementation()["implementation_sha256"]
        ),
        "resume_implementation_sha256": (
            lfm2_cpcf_v5_resume_implementation()["implementation_sha256"]
        ),
        "selector_contract_sha256": (
            _optional_runtime_contract_sha256(
                sealed.backend.candidate_selector,
                kind="candidate_selector",
            )
        ),
        "gpu_contract_sha256": _optional_runtime_contract_sha256(
            sealed.backend.candidate_replay_evaluator,
            kind="candidate_replay",
        ),
    }
    expected["numerical_contract_sha256"] = _optional_runtime_contract_sha256(
        sealed.backend.numerical_frontier_adapter,
        kind="numerical_frontier",
    )
    if (
        state.next_layer_index > sealed.backend.expected_layer_count
        or any(context.get(name) != value for name, value in expected.items())
        or sealed.selected_target_sha256 != canonical_sha256(selected_target.as_dict())
    ):
        raise Lfm2CpcfCatalogError(
            "Admitted V5 resume differs from the exact live parent, cache, "
            "bridge/tokenizer, blank topology, selected plan, runtime "
            "policy, compiler, numerical adapter, or GPU context."
        )


def _resume_sidecar_paths_from_roots(
    roots: Sequence[tuple[str, Any]],
    *,
    forbidden_objects: Mapping[str, Any],
) -> tuple[str, ...]:
    """Find production-resource identities captured by object graphs."""

    torch = _torch()
    forbidden = {
        id(value): name
        for name, value in forbidden_objects.items()
        if value is not None
    }
    found: set[str] = set()
    visited: set[int] = set()

    def walk(value: Any, path: str, depth: int) -> None:
        identity = id(value)
        if identity in forbidden:
            found.add(f"{path}->{forbidden[identity]}")
            return
        if (
            depth >= 8
            or identity in visited
            or isinstance(
                value,
                (
                    str,
                    bytes,
                    bytearray,
                    int,
                    float,
                    bool,
                    type(None),
                    Path,
                    torch.Tensor,
                    type,
                    ModuleType,
                    CodeType,
                ),
            )
        ):
            return
        visited.add(identity)
        if isinstance(value, partial):
            walk(value.func, f"{path}.__partial_func__", depth + 1)
            walk(value.args, f"{path}.__partial_args__", depth + 1)
            walk(value.keywords, f"{path}.__partial_keywords__", depth + 1)
        bound_self = getattr(value, "__self__", None)
        if bound_self is not None:
            walk(bound_self, f"{path}.__self__", depth + 1)
        if isinstance(value, FunctionType):
            closure = value.__closure__
            if closure is not None:
                for index, cell in enumerate(closure):
                    try:
                        child = cell.cell_contents
                    except ValueError:
                        continue
                    walk(child, f"{path}.__closure__[{index}]", depth + 1)
            walk(value.__defaults__, f"{path}.__defaults__", depth + 1)
            walk(value.__kwdefaults__, f"{path}.__kwdefaults__", depth + 1)
            walk(vars(value), f"{path}.__dict__", depth + 1)
            return
        if isinstance(value, Mapping):
            for key, child in value.items():
                walk(child, f"{path}[{key!r}]", depth + 1)
            return
        if isinstance(value, (tuple, list, set, frozenset)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]", depth + 1)
            return
        namespace = getattr(value, "__dict__", None)
        if isinstance(namespace, Mapping):
            for key, child in namespace.items():
                walk(child, f"{path}.{key}", depth + 1)

    for path, value in roots:
        walk(value, path, 0)
    return tuple(sorted(found))


def _model_resume_sidecar_paths(
    model: Any,
    *,
    forbidden_objects: Mapping[str, Any],
) -> tuple[str, ...]:
    """Find exact production-resource identities attached to a child."""

    return _resume_sidecar_paths_from_roots(
        tuple(
            (
                f"module[{module_name or '<root>'}]",
                vars(module),
            )
            for module_name, module in model.named_modules()
        ),
        forbidden_objects=forbidden_objects,
    )


@dataclass(frozen=True, slots=True)
class _V5StandaloneModelLoader:
    """Audit both the loader binding and every model it freshly returns."""

    loader: Callable[[Path], Any]
    forbidden_objects: Mapping[str, Any]
    v5_resource_fingerprint: str
    resource_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        implementation, resource, declared = _callable_fingerprints(
            self.loader,
            name="V5 underlying model_loader",
        )
        body = {
            "schema": "neural-forge-lfm2-cpcf-v5-standalone-loader-v1",
            "v5_resource_fingerprint": _digest(
                self.v5_resource_fingerprint,
                "V5 loader resource fingerprint",
            ),
            "underlying_loader_implementation_fingerprint": implementation,
            "underlying_loader_resource_fingerprint": resource,
            "underlying_loader_declared_resource_fingerprint": declared,
            "forbidden_resource_labels": sorted(self.forbidden_objects),
            "fresh_reload_sidecar_audit_required": True,
        }
        object.__setattr__(
            self,
            "resource_fingerprint",
            canonical_sha256(body),
        )

    def __call__(self, source: Path) -> Any:
        model = self.loader(source)
        paths = _model_resume_sidecar_paths(
            model,
            forbidden_objects=self.forbidden_objects,
        )
        if paths:
            raise Lfm2CpcfCatalogError(
                "Fresh V5 loader attached a parent, cache, coordinator, "
                "checkpoint, pending admission, or resume sidecar to the "
                "standalone child."
            )
        return model


def _named_install_tensors(
    model: Any,
) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (f"parameter.{name}", parameter) for name, parameter in _named_parameters(model)
    ) + tuple((f"buffer.{name}", buffer) for name, buffer in _named_buffers(model))


def _install_staged_candidate(
    *,
    live_model: Any,
    live_snapshot: _CpcfModelSnapshot,
    staging_model: Any,
    staging_snapshot: _CpcfModelSnapshot,
    staging_candidate_state_fingerprint: str,
    staging_parameter_receipt: Mapping[str, Any],
    authenticated_compiler_write_paths: Sequence[str],
    expected_unique_parameters: int,
) -> tuple[str, Mapping[str, Any], Mapping[str, Any]]:
    torch = _torch()
    candidate = _digest(
        staging_candidate_state_fingerprint,
        "staging_candidate_state_fingerprint",
    )
    _assert_model_matches_snapshot_topology(
        live_model,
        live_snapshot,
        name="live blank before staged installation",
    )
    _assert_model_matches_snapshot_topology(
        staging_model,
        staging_snapshot,
        name="staging candidate",
    )
    if (
        fingerprint_model_state(live_model) != live_snapshot.state_fingerprint
        or fingerprint_model_state(staging_model) != candidate
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF live blank or staging candidate changed before "
            "authenticated installation."
        )
    live_values = dict(_named_install_tensors(live_model))
    staging_values = dict(_named_install_tensors(staging_model))
    if (
        len(live_values) != len(_named_install_tensors(live_model))
        or len(staging_values) != len(_named_install_tensors(staging_model))
        or set(live_values) != set(staging_values)
    ):
        raise Lfm2CpcfCatalogError("CPCF staging/live install tensor paths differ.")
    with torch.no_grad():
        copied_objects: set[int] = set()
        for path in sorted(live_values):
            destination = live_values[path]
            source = staging_values[path]
            if (
                tuple(destination.shape) != tuple(source.shape)
                or destination.dtype != source.dtype
                or destination.layout != source.layout
            ):
                raise Lfm2CpcfCatalogError(
                    f"CPCF staged install tensor {path!r} differs in shape, "
                    "dtype, or layout."
                )
            if id(destination) in copied_objects:
                continue
            destination.copy_(
                source.detach().to(
                    device=destination.device,
                    dtype=destination.dtype,
                )
            )
            copied_objects.add(id(destination))
    installed_candidate = fingerprint_model_state(live_model)
    if installed_candidate != candidate:
        raise Lfm2CpcfCatalogError(
            "CPCF controlled staged copy did not reproduce the exact "
            "authenticated candidate state."
        )
    _assert_model_matches_snapshot_topology(
        live_model,
        live_snapshot,
        name="live candidate after staged installation",
    )
    live_parameter_receipt = _parameter_path_write_receipt(
        live_model,
        live_snapshot,
        expected_unique_parameters=expected_unique_parameters,
        candidate_state_fingerprint=installed_candidate,
        authenticated_compiler_write_paths=(authenticated_compiler_write_paths),
    )
    if staging_parameter_receipt.get(
        "resulting_parameter_path_aggregate_sha256"
    ) != live_parameter_receipt.get(
        "resulting_parameter_path_aggregate_sha256"
    ) or staging_parameter_receipt.get(
        "target_unique_parameters"
    ) != live_parameter_receipt.get("target_unique_parameters"):
        raise Lfm2CpcfCatalogError(
            "CPCF live installation differs from its authenticated staging "
            "parameter receipt."
        )
    body = {
        "schema": LFM2_CPCF_STAGED_INSTALL_RECEIPT_SCHEMA,
        "live_blank_state_fingerprint": live_snapshot.state_fingerprint,
        "staging_blank_state_fingerprint": staging_snapshot.state_fingerprint,
        "staging_candidate_state_fingerprint": candidate,
        "installed_candidate_state_fingerprint": installed_candidate,
        "staging_parameter_receipt_sha256": staging_parameter_receipt["receipt_sha256"],
        "live_parameter_receipt_sha256": live_parameter_receipt["receipt_sha256"],
        "copied_tensor_object_count": len(copied_objects),
        "live_parameter_and_buffer_objects_preserved": True,
        "live_parameter_and_buffer_storages_preserved": True,
        "staging_and_live_candidate_states_equal": True,
        "controlled_copy_only": True,
    }
    install_receipt = MappingProxyType(
        {**body, "receipt_sha256": canonical_sha256(body)}
    )
    return installed_candidate, live_parameter_receipt, install_receipt


def _parameter_path_write_receipt(
    model: Any,
    snapshot: _CpcfModelSnapshot,
    *,
    expected_unique_parameters: int,
    candidate_state_fingerprint: str,
    authenticated_compiler_write_paths: Sequence[str],
) -> Mapping[str, Any]:
    """Prove every blank storage was authentically written by the compiler."""

    torch = _torch()
    candidate = _digest(
        candidate_state_fingerprint,
        "candidate_state_fingerprint",
    )
    if fingerprint_model_state(model) != candidate:
        raise Lfm2CpcfCatalogError(
            "CPCF candidate fingerprint changed during path receipt."
        )
    current = _named_parameters(model)
    before_bindings = snapshot.parameter_bindings
    if tuple((name, id(value)) for name, value in current) != tuple(
        (name, id(value)) for name, value, _requires_grad in before_bindings
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF target parameter paths or objects changed during compilation."
        )
    current_storage = {
        name: _parameter_storage_identity(parameter) for name, parameter in current
    }
    if current_storage != dict(snapshot.parameter_storage_identity):
        raise Lfm2CpcfCatalogError(
            "CPCF target parameter storage topology changed during compilation."
        )
    before_by_path = {
        name: snapshot.tensor_values[id(parameter)]
        for name, parameter, _requires_grad in before_bindings
    }
    current_by_path = dict(current)
    if len(current_by_path) != len(current):
        raise Lfm2CpcfCatalogError(
            "CPCF target exposes duplicate parameter path names."
        )
    authenticated_paths = {
        _text(path, "authenticated_compiler_write_path")
        for path in authenticated_compiler_write_paths
    }
    if authenticated_paths != set(current_by_path):
        raise Lfm2CpcfCatalogError(
            "CPCF authenticated compiler write paths do not cover the exact "
            "target parameter topology."
        )
    storage_groups: dict[tuple[Any, ...], list[str]] = {}
    for path, identity in snapshot.parameter_storage_identity.items():
        storage_groups.setdefault(identity, []).append(path)
    group_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    unique_numel = 0
    changed_storage_count = 0
    zero_valued_folded_storage_count = 0
    for identity, raw_paths in sorted(
        storage_groups.items(),
        key=lambda item: sorted(item[1]),
    ):
        paths = sorted(raw_paths)
        representative = current_by_path[paths[0]]
        unique_numel += int(representative.numel())
        before_hashes = {
            _tensor_sha256(
                before_by_path[path],
                f"compiler blank parameter {path}",
            )
            for path in paths
        }
        after_hashes = {
            _tensor_sha256(
                current_by_path[path],
                f"compiled parameter {path}",
            )
            for path in paths
        }
        if len(before_hashes) != 1 or len(after_hashes) != 1:
            raise Lfm2CpcfCatalogError(
                "CPCF tied parameter aliases disagree on before/after bytes."
            )
        before_sha = before_hashes.pop()
        after_sha = after_hashes.pop()
        before_values = before_by_path[paths[0]]
        group_written = bool(set(paths).intersection(authenticated_paths))
        if int(torch.count_nonzero(before_values).item()) != 0 or not group_written:
            raise Lfm2CpcfCatalogError(
                "CPCF did not authentically write every unique target "
                "parameter storage from the compiler-zero blank."
            )
        changed = before_sha != after_sha
        changed_storage_count += int(changed)
        zero_valued = int(torch.count_nonzero(representative.detach()).item()) == 0
        zero_valued_folded_storage_count += int(zero_valued)
        alias_group_sha = canonical_sha256(
            {
                "paths": paths,
                "shape": [int(size) for size in representative.shape],
                "dtype": str(representative.dtype).removeprefix("torch."),
                "numel": int(representative.numel()),
            }
        )
        group_rows.append(
            {
                "alias_group_sha256": alias_group_sha,
                "paths": paths,
                "tied_alias": len(paths) > 1,
                "shape": [int(size) for size in representative.shape],
                "dtype": str(representative.dtype).removeprefix("torch."),
                "numel": int(representative.numel()),
                "compiler_zero_sha256": before_sha,
                "result_sha256": after_sha,
                "changed_from_compiler_zero": changed,
                "authenticated_compiler_write": True,
                "zero_valued_folded_result": zero_valued,
                "overwritten_from_compiler_blank": True,
            }
        )
        path_rows.extend(
            {
                "parameter_path": path,
                "alias_group_sha256": alias_group_sha,
                "compiler_zero_sha256": before_sha,
                "result_sha256": after_sha,
                "changed_from_compiler_zero": changed,
                "authenticated_compiler_write": True,
                "overwritten_from_compiler_blank": True,
            }
            for path in paths
        )
    if unique_numel != expected_unique_parameters:
        raise Lfm2CpcfCatalogError(
            "CPCF independent parameter-path inventory differs from the "
            "exact target budget."
        )
    path_rows.sort(key=lambda row: str(row["parameter_path"]))
    group_rows.sort(key=lambda row: str(row["alias_group_sha256"]))
    result_aggregate = canonical_sha256(
        [
            {
                "parameter_path": row["parameter_path"],
                "alias_group_sha256": row["alias_group_sha256"],
                "result_sha256": row["result_sha256"],
            }
            for row in path_rows
        ]
    )
    body = {
        "schema": LFM2_CPCF_PARAMETER_PATH_WRITE_RECEIPT_SCHEMA,
        "compiler_blank_state_fingerprint": snapshot.state_fingerprint,
        "candidate_state_fingerprint": candidate,
        "named_parameter_path_count": len(path_rows),
        "unique_parameter_storage_count": len(group_rows),
        "target_unique_parameters": unique_numel,
        "parameter_paths": path_rows,
        "alias_groups": group_rows,
        "resulting_parameter_path_aggregate_sha256": result_aggregate,
        "all_named_parameter_paths_accounted": True,
        "authenticated_compiler_write_paths": sorted(authenticated_paths),
        "changed_unique_parameter_storage_count": changed_storage_count,
        "zero_valued_folded_storage_count": (zero_valued_folded_storage_count),
        "all_unique_parameter_storages_changed_from_compiler_zero": (
            changed_storage_count == len(group_rows)
        ),
        "all_unique_parameter_storages_authentically_written": True,
        "resulting_path_hashes_verified": True,
        "tied_aliases_explicitly_accounted": True,
    }
    return MappingProxyType({**body, "receipt_sha256": canonical_sha256(body)})


def _verify_source_tied_table(
    parent_model: Any,
    table: Any,
    trace_adapter: Lfm2StrictTokenizerBridgeTraceAdapter,
) -> None:
    torch = _torch()
    get_embedding = getattr(parent_model, "get_input_embeddings", None)
    embedding = get_embedding() if callable(get_embedding) else None
    weight = getattr(embedding, "weight", None)
    mapping = tuple(int(value) for value in trace_adapter.mapping)
    if (
        not isinstance(weight, torch.Tensor)
        or weight.ndim != 2
        or tuple(table.shape) != (len(mapping), int(weight.shape[1]))
        or any(value < 0 or value >= int(weight.shape[0]) for value in mapping)
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF tied interface table shape or bridge rows differ from the "
            "live parent embedding."
        )
    with torch.no_grad():
        for start in range(0, len(mapping), 1024):
            stop = min(len(mapping), start + 1024)
            indices = torch.tensor(
                mapping[start:stop],
                device=weight.device,
                dtype=torch.long,
            )
            expected = weight.detach().index_select(0, indices)
            observed = (
                table[start:stop]
                .detach()
                .to(
                    device=weight.device,
                    dtype=weight.dtype,
                )
            )
            if not torch.equal(expected, observed):
                raise Lfm2CpcfCatalogError(
                    "CPCF source tied table is not the bridge-selected live "
                    "parent embedding."
                )


def _authenticated_compiler_write_paths(
    report: Mapping[str, Any],
) -> tuple[str, ...]:
    guard = report.get("compile_sealed_storage_write_guard")
    paths = (
        None
        if not isinstance(guard, Mapping)
        else guard.get("authenticated_writable_parameter_paths")
    )
    if (
        not isinstance(paths, list)
        or not paths
        or any(not isinstance(path, str) or not path for path in paths)
        or paths != sorted(set(paths))
        or guard.get("all_unique_writable_parameter_storages_written") is not True
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF report lacks its authenticated staging parameter write set."
        )
    return tuple(paths)


def _authenticate_nested_seal(
    value: Any,
    *,
    seal_name: str,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2CpcfCatalogError(f"{name} must be a sealed mapping.")
    body = deepcopy(dict(value))
    supplied = _digest(body.pop(seal_name, None), f"{name}.{seal_name}")
    if canonical_sha256(body) != supplied:
        raise Lfm2CpcfCatalogError(f"{name} seal is invalid.")
    return MappingProxyType({**body, seal_name: supplied})


def _authenticate_v5_admission_proof_compiler_binding(
    chain_proof: Mapping[str, Any],
    *,
    compiler_report: Mapping[str, Any],
    implementation: Mapping[str, Any],
    layer_count: int,
) -> None:
    """Bind a standalone admission proof to this exact compiler execution."""

    proof_frontier = chain_proof.get("boundary_reports")
    proof_context = (
        None
        if not isinstance(proof_frontier, Mapping)
        else proof_frontier.get("context")
    )
    compiler_policy = compiler_report.get("policy")
    if (
        not isinstance(proof_frontier, Mapping)
        or not isinstance(proof_context, Mapping)
        or not isinstance(compiler_policy, Mapping)
    ):
        raise Lfm2CpcfCatalogError(
            "V5 admission-chain proof lacks compiler-bound frontier context."
        )
    try:
        physical_rss_policy_sha256 = build_lfm2_cpcf_v5_physical_rss_policy_sha256(
            full_phase_tree_logical_bytes=compiler_policy[
                "full_phase_tree_logical_bytes"
            ],
            minimum_reduction_fraction=compiler_policy[
                "minimum_physical_rss_reduction_fraction"
            ],
            maximum_increase_bytes=compiler_policy[
                "maximum_physical_rss_increase_bytes"
            ],
            maximum_process_rss_bytes=compiler_policy["maximum_process_rss_bytes"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Lfm2CpcfCatalogError(
            "V5 compiler policy cannot authenticate an admission-chain proof."
        ) from exc
    if (
        proof_frontier.get("expected_layer_count") != layer_count
        or proof_frontier.get("ordered_phase_sha256")
        != compiler_report.get("phase_sha256")
        or proof_context.get("selective_catalog_sha256")
        != compiler_report.get("selective_catalog_sha256")
        or proof_context.get("selective_source_sha256")
        != compiler_report.get("selective_source_sha256")
        or proof_context.get("selective_provider_sha256")
        != compiler_report.get("selective_provider_sha256")
        or proof_context.get("runtime_policy_sha256")
        != canonical_sha256(compiler_policy)
        or proof_context.get("physical_rss_policy_sha256") != physical_rss_policy_sha256
        or proof_context.get("compiler_implementation_sha256")
        != implementation["implementation_sha256"]
        or proof_context.get("resume_implementation_sha256")
        != lfm2_cpcf_v5_resume_implementation()["implementation_sha256"]
        or proof_context.get("bootstrap_authentication_sha256")
        != compiler_report.get("bootstrap_authentication_sha256")
        or proof_context.get("final_norm_authentication_sha256")
        != compiler_report.get("final_norm_authentication_sha256")
    ):
        raise Lfm2CpcfCatalogError(
            "V5 admission-chain proof belongs to a different compiler execution."
        )


def _authenticate_v5_model_fold_report(
    compiler_report: Mapping[str, Any],
    *,
    backend: Lfm2CpcfV5SelectiveBackend | None,
    resume_contract: Mapping[str, Any] | None = None,
    resume_coordinator: Lfm2CpcfV5ResumeCoordinator | None = None,
) -> None:
    """Authenticate stable v5 evidence while allowing sampled RSS values."""

    operation_audit = compiler_report.get("operation_audit")
    required_zero_audit = {
        "parent_model_forward_calls": 0,
        "target_model_forward_calls": 0,
        "generation_calls": 0,
        "parent_cache_generation_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "fallback_paths": 0,
    }
    implementation = _authenticate_nested_seal(
        compiler_report.get("compiler_implementation"),
        seal_name="implementation_sha256",
        name="v5 compiler implementation",
    )
    live_implementation = dict(lfm2_cpcf_v5_compiler_implementation())
    residency = _authenticate_nested_seal(
        compiler_report.get("residency_receipt"),
        seal_name="snapshot_sha256",
        name="v5 residency receipt",
    )
    physical = _authenticate_nested_seal(
        compiler_report.get("physical_rss_receipt"),
        seal_name="report_sha256",
        name="v5 physical RSS receipt",
    )
    short_conv_layers = compiler_report.get("short_conv_layer_indices")
    selector_layers = compiler_report.get("preinstall_candidate_selector_layer_indices")
    numerical_layers = compiler_report.get(
        "short_conv_v5_real_numerical_provenance_layer_indices"
    )
    resume_publication = compiler_report.get("resume_publication_enabled")
    resume_used = compiler_report.get("resume_used")
    resume_start = compiler_report.get("resume_start_layer_index")
    ordered_layer_results = compiler_report.get("ordered_layer_result_sha256")
    layer_count = (
        -1
        if not isinstance(ordered_layer_results, list)
        else len(ordered_layer_results)
    )
    admitted_prefixes = compiler_report.get("resume_admitted_prefixes")
    if (
        not isinstance(resume_publication, bool)
        or not isinstance(resume_used, bool)
        or resume_used
        and not resume_publication
        or isinstance(resume_start, bool)
        or not isinstance(resume_start, int)
        or resume_start < 0
        or layer_count < 0
        or resume_start > layer_count
        or compiler_report.get("resume_completed_prefix_layer_indices")
        != list(range(resume_start))
        or compiler_report.get("resume_executed_layer_indices")
        != list(range(resume_start, layer_count))
        or compiler_report.get("resume_completed_layers_replayed") is not False
        or compiler_report.get("resume_actual_upstream_child_state_restored")
        is not resume_used
        or compiler_report.get("resume_checkpoint_receipt_chain_authenticated")
        is not True
        or compiler_report.get("resume_parent_or_cache_tensor_sidecars") != 0
        or not isinstance(admitted_prefixes, list)
    ):
        raise Lfm2CpcfCatalogError(
            "V5 compiled report has an invalid resume execution or "
            "publication state machine."
        )
    for digest_field in (
        "resume_prefix_install_report_sha256",
        "resume_frontier_sha256",
        "resume_admission_sha256",
    ):
        value = compiler_report.get(digest_field)
        if resume_used:
            _digest(value, f"v5 compiler {digest_field}")
        elif value is not None:
            raise Lfm2CpcfCatalogError(
                f"Cold V5 report unexpectedly populated {digest_field}."
            )
    input_proof_value = compiler_report.get("resume_input_admission_chain_proof")
    input_proof_sha256_value = compiler_report.get(
        "resume_input_admission_chain_proof_sha256"
    )
    if resume_used:
        try:
            input_proof = dict(
                authenticate_lfm2_cpcf_v5_admission_chain_proof(input_proof_value)
            )
        except Exception as exc:
            raise Lfm2CpcfCatalogError(
                "Resumed V5 compiler report has no valid standalone input "
                "admission-chain proof."
            ) from exc
        input_proof_sha256 = _digest(
            input_proof_sha256_value,
            "resume input admission-chain proof SHA-256",
        )
        input_pending = input_proof["pending_document"]
        input_admission = input_proof["admission_document"]
        _authenticate_v5_admission_proof_compiler_binding(
            input_proof,
            compiler_report=compiler_report,
            implementation=implementation,
            layer_count=layer_count,
        )
        if (
            input_proof["proof_sha256"] != input_proof_sha256
            or input_pending.get("next_layer_index") != resume_start
            or input_pending.get("frontier_sha256")
            != compiler_report.get("resume_frontier_sha256")
            or input_admission.get("admission_sha256")
            != compiler_report.get("resume_admission_sha256")
        ):
            raise Lfm2CpcfCatalogError(
                "Resumed V5 compiler input differs from its standalone "
                "admission-chain proof."
            )
    elif input_proof_value is not None or input_proof_sha256_value is not None:
        raise Lfm2CpcfCatalogError(
            "Cold V5 compiler report unexpectedly carries an input "
            "admission-chain proof."
        )
    expected_admitted_indices = (
        list(
            range(
                resume_start + (1 if resume_used else 0),
                layer_count + 1,
            )
        )
        if resume_publication
        else []
    )
    if [
        record.get("next_layer_index")
        for record in admitted_prefixes
        if isinstance(record, Mapping)
    ] != expected_admitted_indices or len(admitted_prefixes) != len(
        expected_admitted_indices
    ):
        raise Lfm2CpcfCatalogError(
            "V5 admitted-prefix inventory is not the exact executed publication suffix."
        )
    if resume_publication and type(resume_coordinator) is not (
        Lfm2CpcfV5ResumeCoordinator
    ):
        raise Lfm2CpcfCatalogError(
            "V5 resume publication must be authenticated through its exact "
            "bound coordinator and artifact store."
        )
    for record in admitted_prefixes:
        base_record_fields = {
            "next_layer_index",
            "frontier_reference",
            "checkpoint_reference",
            "frontier_sha256",
            "admission_sha256",
        }
        proof_fields = {
            "admission_chain_proof",
            "admission_chain_proof_sha256",
        }
        proof_present = isinstance(record, Mapping) and bool(
            set(record).intersection(proof_fields)
        )
        if (
            not isinstance(record, Mapping)
            or not proof_present
            or set(record)
            != base_record_fields | (proof_fields if proof_present else set())
            or not isinstance(record["frontier_reference"], Mapping)
            or not isinstance(record["checkpoint_reference"], Mapping)
        ):
            raise Lfm2CpcfCatalogError("V5 admitted-prefix record is incomplete.")
        try:
            frontier_reference = Lfm2CpcfArtifactReference(
                **dict(record["frontier_reference"])
            )
            checkpoint_reference = Lfm2CpcfArtifactReference(
                **dict(record["checkpoint_reference"])
            )
        except Exception as exc:
            raise Lfm2CpcfCatalogError(
                "V5 admitted-prefix record has invalid typed artifact references."
            ) from exc
        frontier_sha256 = _digest(
            record["frontier_sha256"],
            "admitted frontier SHA-256",
        )
        admission_sha256 = _digest(
            record["admission_sha256"],
            "admitted admission SHA-256",
        )
        if (
            frontier_reference.artifact_kind != "boundary"
            or checkpoint_reference.artifact_kind != "checkpoint"
        ):
            raise Lfm2CpcfCatalogError(
                "V5 admitted-prefix artifact reference kinds are invalid."
            )
        if proof_present:
            try:
                chain_proof = dict(
                    authenticate_lfm2_cpcf_v5_admission_chain_proof(
                        record["admission_chain_proof"]
                    )
                )
            except Exception as exc:
                raise Lfm2CpcfCatalogError(
                    "V5 admitted-prefix standalone admission-chain proof is invalid."
                ) from exc
            proof_sha256 = _digest(
                record["admission_chain_proof_sha256"],
                "admitted admission-chain proof SHA-256",
            )
            proof_pending = chain_proof["pending_document"]
            proof_admission = chain_proof["admission_document"]
            _authenticate_v5_admission_proof_compiler_binding(
                chain_proof,
                compiler_report=compiler_report,
                implementation=implementation,
                layer_count=layer_count,
            )
            if (
                chain_proof["proof_sha256"] != proof_sha256
                or proof_pending.get("next_layer_index") != record["next_layer_index"]
                or proof_pending.get("frontier_reference")
                != frontier_reference.as_dict()
                or proof_pending.get("checkpoint_reference")
                != checkpoint_reference.as_dict()
                or proof_pending.get("frontier_sha256") != frontier_sha256
                or proof_admission.get("admission_sha256") != admission_sha256
            ):
                raise Lfm2CpcfCatalogError(
                    "V5 admitted-prefix record differs from its standalone "
                    "admission-chain proof."
                )
        if resume_coordinator is not None:
            try:
                reopened = resume_coordinator.open_admitted(admission_sha256)
            except Exception as exc:
                raise Lfm2CpcfCatalogError(
                    "V5 admitted-prefix record did not reopen through its "
                    "bound coordinator and artifact store."
                ) from exc
            if (
                reopened.next_layer_index != record["next_layer_index"]
                or reopened.frontier_reference != frontier_reference
                or reopened.checkpoint_reference != checkpoint_reference
                or reopened.frontier.frontier_sha256 != frontier_sha256
                or reopened.admission.get("admission_sha256") != admission_sha256
            ):
                raise Lfm2CpcfCatalogError(
                    "V5 admitted-prefix record differs from the exact "
                    "reauthenticated coordinator/store document chain."
                )
    if resume_used and resume_coordinator is not None:
        resume_admission_sha256 = _digest(
            compiler_report.get("resume_admission_sha256"),
            "resume input admission SHA-256",
        )
        try:
            reopened_input = resume_coordinator.open_admitted(resume_admission_sha256)
        except Exception as exc:
            raise Lfm2CpcfCatalogError(
                "V5 resumed input prefix did not reopen through its bound "
                "coordinator and artifact store."
            ) from exc
        if (
            reopened_input.next_layer_index != resume_start
            or reopened_input.frontier.frontier_sha256
            != compiler_report.get("resume_frontier_sha256")
        ):
            raise Lfm2CpcfCatalogError(
                "V5 resumed input prefix differs from its exact "
                "reauthenticated coordinator/store state."
            )
    if resume_contract is not None:
        contract = _plain_json(resume_contract)
        expected_contract_fields = {
            "schema",
            "resume_publication_enabled",
            "resume_used",
            "coordinator_store_binding_sha256",
            "admission_sha256",
            "next_layer_index",
            "frontier_sha256",
            "context_sha256",
            "frontier_reference",
            "checkpoint_reference",
        }
        if (
            not isinstance(contract, dict)
            or set(contract) != expected_contract_fields
            or contract.get("schema") != LFM2_CPCF_V5_RESUME_SCHEMA
            or contract.get("resume_publication_enabled") is not resume_publication
            or contract.get("resume_used") is not resume_used
            or (
                resume_publication
                and _digest(
                    contract.get("coordinator_store_binding_sha256"),
                    "resume coordinator store binding",
                )
                != contract["coordinator_store_binding_sha256"]
            )
            or (
                not resume_publication
                and contract.get("coordinator_store_binding_sha256") is not None
            )
        ):
            raise Lfm2CpcfCatalogError(
                "V5 compiler resume report differs from its sealed stack "
                "resume contract."
            )
        if resume_used:
            if (
                contract.get("next_layer_index") != resume_start
                or contract.get("admission_sha256")
                != compiler_report.get("resume_admission_sha256")
                or contract.get("frontier_sha256")
                != compiler_report.get("resume_frontier_sha256")
                or not isinstance(contract.get("frontier_reference"), dict)
                or not isinstance(contract.get("checkpoint_reference"), dict)
                or contract.get("frontier_reference")
                != input_pending.get("frontier_reference")
                or contract.get("checkpoint_reference")
                != input_pending.get("checkpoint_reference")
            ):
                raise Lfm2CpcfCatalogError(
                    "Resumed V5 compiler report differs from its admitted "
                    "input prefix contract."
                )
            _digest(contract["context_sha256"], "resume context SHA-256")
        elif any(
            contract.get(field) is not None
            for field in (
                "admission_sha256",
                "next_layer_index",
                "frontier_sha256",
                "context_sha256",
                "frontier_reference",
                "checkpoint_reference",
            )
        ):
            raise Lfm2CpcfCatalogError(
                "Cold V5 stack resume contract contains continuation state."
            )
    if (
        dict(implementation) != live_implementation
        or compiler_report.get("resume_supported") is not True
        or compiler_report.get("resume_fail_closed") is not True
        or compiler_report.get("all_layers_compiled_in_order") is not True
        or compiler_report.get("actual_folded_upstream_replay_used") is not True
        or compiler_report.get("source_state_preserved") is not True
        or compiler_report.get("target_parameter_count_preserved") is not True
        or compiler_report.get("full_all_layer_phase_tree_materialized") is not False
        or compiler_report.get("tensor_free_phase_shells_used") is not True
        or compiler_report.get(
            "active_layer_evidence_released_before_successor_compile"
        )
        is not True
        or compiler_report.get(
            "final_norm_evidence_separately_bounded_and_reauthenticated"
        )
        is not True
        or compiler_report.get("candidate_replay_backend_injected") is not True
        or compiler_report.get("short_conv_v5_endpoint_first_selector_integrated")
        is not True
        or compiler_report.get("short_conv_v5_real_numerical_adapter_required")
        is not True
        or compiler_report.get(
            "short_conv_v5_real_numerical_adapter_concrete_instance_provided"
        )
        is not True
        or compiler_report.get(
            "short_conv_v5_numerical_candidate_generation_integrated"
        )
        is not True
        or compiler_report.get("short_conv_v5_numerical_policy_integrated") is not True
        or compiler_report.get("short_conv_v5_post_install_rescoring_performed")
        is not False
        or compiler_report.get("legacy_v4_default_or_fingerprint_changed") is not False
        or not isinstance(operation_audit, Mapping)
        or dict(operation_audit) != required_zero_audit
        or not isinstance(short_conv_layers, list)
        or not short_conv_layers
        or selector_layers != short_conv_layers
        or numerical_layers != short_conv_layers
        or residency.get("max_resident_layers") != 2
        or residency.get("prefetch_distance_layers") != 1
        or residency.get("resident_bytes") != 0
        or residency.get("peak_resident_bytes", 1)
        > residency.get("maximum_resident_bytes", 0)
        or physical.get("schema") != LFM2_CPCF_V5_PHYSICAL_RSS_SCHEMA
        or physical.get("sampler_stopped") is not True
        or physical.get("physical_rss_gate_passed") is not True
        or physical.get("peak_increase_bytes", 1)
        > physical.get("maximum_increase_bytes", 0)
    ):
        raise Lfm2CpcfCatalogError(
            "V5 compiled report lacks its selective-residency, physical-RSS, "
            "real numerical, endpoint-selection, GPU injection, resume, or "
            "zero-operation invariants."
        )
    if backend is None:
        return
    backend.verify()
    phases = compiler_report.get("phase_sha256")
    if (
        compiler_report.get("policy") != backend.runtime_policy.as_dict()
        or compiler_report.get("selective_catalog_sha256")
        != backend.expected_selective_catalog_sha256
        or compiler_report.get("selective_source_sha256")
        != backend.expected_selective_source_sha256
        or compiler_report.get("selective_provider_sha256")
        != backend.expected_selective_provider_sha256
        or phases != list(backend.expected_phase_sha256)
        or compiler_report.get("phase_count") != len(backend.expected_phase_sha256)
        or not isinstance(
            compiler_report.get("ordered_layer_result_sha256"),
            list,
        )
        or len(compiler_report["ordered_layer_result_sha256"])
        != backend.expected_layer_count
        or compiler_report.get("authenticated_full_phase_tree_logical_bytes")
        != backend.runtime_policy.full_phase_tree_logical_bytes
        or residency.get("catalog_sha256") != backend.expected_selective_catalog_sha256
        or residency.get("source_sha256") != backend.expected_selective_source_sha256
        or residency.get("provider_sha256")
        != backend.expected_selective_provider_sha256
        or residency.get("maximum_resident_bytes")
        != backend.runtime_policy.maximum_resident_evidence_bytes
        or physical.get("full_phase_tree_logical_bytes")
        != backend.runtime_policy.full_phase_tree_logical_bytes
        or physical.get("maximum_increase_bytes")
        != backend.runtime_policy.maximum_physical_rss_increase_bytes
    ):
        raise Lfm2CpcfCatalogError(
            "V5 compiled report differs from its preregistered compiler "
            "policy, selective catalog, ordered phases, or memory bounds."
        )


def _authenticate_compiled_model_fold_report(
    report: Mapping[str, Any],
    *,
    expected_result_sha256: str | None = None,
    v5_backend: Lfm2CpcfV5SelectiveBackend | None = None,
    v5_resume_contract: Mapping[str, Any] | None = None,
    v5_resume_coordinator: Lfm2CpcfV5ResumeCoordinator | None = None,
) -> Mapping[str, Any]:
    """Independently authenticate the compiler-owned immutable report."""

    if not isinstance(report, Mapping):
        raise Lfm2CpcfCatalogError(
            "CPCF compiler returned no model-fold report mapping."
        )
    compiler_report = deepcopy(dict(report))
    result_sha256 = _digest(
        compiler_report.pop("result_sha256", None),
        "compiled_model_fold.result_sha256",
    )
    if (
        expected_result_sha256 is not None
        and _digest(
            expected_result_sha256,
            "compiled_model_fold expected result_sha256",
        )
        != result_sha256
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF compiler result object and serialized report seals differ."
        )
    if canonical_sha256(compiler_report) != result_sha256:
        raise Lfm2CpcfCatalogError(
            "CPCF compiled model-fold result_sha256 seal is invalid."
        )
    schema = compiler_report.get("schema")
    if schema == LFM2_CPCF_V5_MODEL_FOLD_SCHEMA:
        _authenticate_v5_model_fold_report(
            compiler_report,
            backend=v5_backend,
            resume_contract=v5_resume_contract,
            resume_coordinator=v5_resume_coordinator,
        )
    elif (
        schema != LFM2_CPCF_MODEL_FOLD_SCHEMA
        or compiler_report.get("all_layers_compiled_in_order") is not True
        or compiler_report.get("actual_folded_upstream_replay_used") is not True
        or compiler_report.get("source_state_preserved") is not True
        or compiler_report.get("target_parameter_count_preserved") is not True
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF compiled report lacks complete model-fold invariants."
        )
    return MappingProxyType({**compiler_report, "result_sha256": result_sha256})


def _authenticate_v5_compiler_result(
    result: Any,
    compiler_report: Mapping[str, Any],
    backend: Lfm2CpcfV5SelectiveBackend,
) -> None:
    """Deeply bind the v5 top-level report to exact typed layer receipts."""

    if type(result) is not Lfm2CpcfCompiledModelFold:
        raise Lfm2CpcfCatalogError(
            "V5 compiler must return the exact typed model-fold result."
        )
    layers = tuple(result.layer_results)
    if (
        len(layers) != backend.expected_layer_count
        or any(type(receipt) is not Lfm2CpcfCompletedLayerReceipt for receipt in layers)
        or tuple(receipt.layer_index for receipt in layers)
        != tuple(range(backend.expected_layer_count))
        or compiler_report.get("ordered_layer_result_sha256")
        != [receipt.result_sha256 for receipt in layers]
        or dict(result.as_dict()) != dict(compiler_report)
    ):
        raise Lfm2CpcfCatalogError(
            "V5 compiler result is not the exact ordered typed layer set "
            "sealed by its top-level report."
        )
    contract = _authenticate_nested_seal(
        compiler_report.get("short_conv_v5_real_numerical_adapter_contract"),
        seal_name="contract_sha256",
        name="v5 real numerical adapter contract",
    )
    live_contract = _real_v5_numerical_adapter_contract(
        backend.numerical_frontier_adapter
    )
    proofs = compiler_report.get("short_conv_v5_real_numerical_layer_proofs")
    short_conv_receipts = tuple(
        receipt for receipt in layers if receipt.operator_kind == "short_conv"
    )
    if (
        not isinstance(live_contract, Mapping)
        or dict(contract) != dict(live_contract)
        or not isinstance(proofs, list)
        or len(proofs) != len(short_conv_receipts)
        or not short_conv_receipts
        or compiler_report.get("short_conv_layer_indices")
        != [receipt.layer_index for receipt in short_conv_receipts]
        or compiler_report.get("short_conv_v5_real_numerical_provenance_layer_indices")
        != [receipt.layer_index for receipt in short_conv_receipts]
    ):
        raise Lfm2CpcfCatalogError(
            "V5 compiler report is not bound to the live real numerical "
            "adapter and every typed short-conv layer."
        )
    recomputed_proofs = []
    for receipt in short_conv_receipts:
        proof = _prove_real_v5_numerical_layer(
            receipt,
            adapter_contract=live_contract,
        )
        if not isinstance(proof, Mapping):
            raise Lfm2CpcfCatalogError(
                "A typed short-conv receipt lacks concrete numerical fold provenance."
            )
        recomputed_proofs.append(dict(proof))
    if proofs != recomputed_proofs:
        raise Lfm2CpcfCatalogError(
            "V5 numerical layer proofs differ from independently "
            "recomputed typed receipt proofs."
        )


def _walk_nested_mappings(value: Any) -> tuple[Mapping[str, Any], ...]:
    result: list[Mapping[str, Any]] = []
    pending = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(current, Mapping):
            result.append(current)
            pending.extend(current.values())
        elif isinstance(current, Sequence) and not isinstance(
            current,
            (str, bytes, bytearray),
        ):
            pending.extend(current)
    return tuple(result)


def _authenticate_v5_gpu_execution_result(
    result: Any,
    backend: Lfm2CpcfV5SelectiveBackend,
) -> Mapping[str, Any]:
    """Prove the current compile, not only a prior receipt, used CUDA replay."""

    layer_results = getattr(result, "layer_results", None)
    if not isinstance(layer_results, tuple) or not layer_results:
        raise Lfm2CpcfCatalogError("V5 compiler result has no typed layer receipts.")
    executions = [
        mapping
        for layer in layer_results
        for mapping in _walk_nested_mappings(layer.as_dict())
        if mapping.get("schema") == DYNAMIC_REPLAY_EXECUTION_SCHEMA
    ]
    device_indices = {
        int(replica.device.device_index)
        for replica in backend.candidate_replay_evaluator.scheduler.replicas
    }
    execution_sha256 = []
    for index, execution in enumerate(executions):
        authenticated = _authenticate_nested_seal(
            execution,
            seal_name="report_sha256",
            name=f"v5 GPU execution[{index}]",
        )
        parity_devices = {
            int(row.get("device_index"))
            for row in authenticated.get("parity_reports", ())
            if isinstance(row, Mapping)
        }
        scheduled = authenticated.get("scheduled_records")
        if (
            parity_devices != device_indices
            or authenticated.get("parity_passed_before_scheduled_work") is not True
            or authenticated.get("backend_receipts_reauthenticated") is not True
            or authenticated.get("installed_state_reauthenticated") is not True
            or authenticated.get("cpu_reference_state_reauthenticated") is not True
            or authenticated.get("canonical_result_order_restored") is not True
            or not isinstance(scheduled, list)
            or not scheduled
            or any(not isinstance(row, Mapping) for row in scheduled)
            or any(
                int(row.get("device_index")) not in device_indices for row in scheduled
            )
        ):
            raise Lfm2CpcfCatalogError(
                "Current v5 candidate replay lacks authenticated two-GPU "
                "parity, scheduled CUDA work, or runtime/state "
                "reauthentication."
            )
        execution_sha256.append(authenticated["report_sha256"])
    if not executions:
        raise Lfm2CpcfCatalogError(
            "V5 production compile injected a GPU object but scheduled no "
            "authenticated CUDA candidate replay."
        )
    body = {
        "schema": LFM2_CPCF_V5_GPU_EXECUTION_BINDING_SCHEMA,
        "backend_fingerprint": backend.backend_fingerprint,
        "gpu_backend_report_sha256": (backend.gpu_backend_report["report_sha256"]),
        "execution_count": len(execution_sha256),
        "execution_report_sha256": execution_sha256,
        "logical_device_indices": sorted(device_indices),
        "two_gpu_parity_reauthenticated_per_execution": True,
        "scheduled_cuda_candidate_replay_proven": True,
    }
    return MappingProxyType({**body, "binding_sha256": canonical_sha256(body)})


def _seal_catalog_compile_report(
    body: Mapping[str, Any],
    *,
    compiled_model_fold_result_sha256: str,
) -> Mapping[str, Any]:
    """Seal adapter/install metadata without modifying compiler provenance."""

    normalized = deepcopy(dict(body))
    if "report_sha256" in normalized:
        raise Lfm2CpcfCatalogError("CPCF catalog compile-report body must be unsealed.")
    normalized["schema"] = LFM2_CPCF_CATALOG_COMPILE_REPORT_SCHEMA
    normalized["compiled_model_fold_result_sha256"] = _digest(
        compiled_model_fold_result_sha256,
        "compiled_model_fold_result_sha256",
    )
    return MappingProxyType(
        {**normalized, "report_sha256": canonical_sha256(normalized)}
    )


def _authenticate_catalog_compile_report(
    report: Mapping[str, Any],
    *,
    compiled_model_fold_result_sha256: str,
    expected_report_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Authenticate the separately sealed catalog/install wrapper."""

    if not isinstance(report, Mapping):
        raise Lfm2CpcfCatalogError("CPCF catalog compile report must be a mapping.")
    catalog_report = deepcopy(dict(report))
    report_sha256 = _digest(
        catalog_report.pop("report_sha256", None),
        "catalog_compile_report.report_sha256",
    )
    compiler_sha256 = _digest(
        catalog_report.get("compiled_model_fold_result_sha256"),
        "catalog_compile_report.compiled_model_fold_result_sha256",
    )
    if expected_report_sha256 is not None and report_sha256 != _digest(
        expected_report_sha256,
        "expected catalog_compile_report.report_sha256",
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF catalog compile report no longer matches its bound seal."
        )
    if (
        catalog_report.get("schema") != LFM2_CPCF_CATALOG_COMPILE_REPORT_SCHEMA
        or compiler_sha256
        != _digest(
            compiled_model_fold_result_sha256,
            "compiled_model_fold_result_sha256",
        )
        or canonical_sha256(catalog_report) != report_sha256
    ):
        raise Lfm2CpcfCatalogError(
            "CPCF catalog compile-report seal or compiler binding is invalid."
        )
    return MappingProxyType({**catalog_report, "report_sha256": report_sha256})


class Lfm2CpcfModelTransaction:
    """Exact transaction around one complete CPCF child installation."""

    method_id = DENSE_BUDGET_MODEL_METHOD

    def __init__(
        self,
        context: FoldExecutionContext,
        snapshot: _CpcfModelSnapshot,
        *,
        v5_backend: Lfm2CpcfV5SelectiveBackend | None = None,
        v5_resume_contract: Mapping[str, Any] | None = None,
        v5_resume_coordinator: Lfm2CpcfV5ResumeCoordinator | None = None,
    ) -> None:
        self.context = context
        self._snapshot = snapshot
        self._v5_backend = v5_backend
        self._v5_resume_contract = v5_resume_contract
        self._v5_resume_coordinator = v5_resume_coordinator
        self.incumbent_state_fingerprint = snapshot.state_fingerprint
        self.candidate_state_fingerprint = snapshot.state_fingerprint
        self._compiled_model_fold_report_canonical = _canonical_json({})
        self._catalog_compile_report_canonical = _canonical_json(
            {
                "schema": LFM2_CPCF_CATALOG_SCHEMA,
                "prepared": False,
            }
        )
        self._catalog_compile_report_sha256: str | None = None
        self.active = True

    @property
    def compiled_model_fold_report(self) -> Mapping[str, Any]:
        """Return a detached view of the immutable compiler report bytes."""

        return MappingProxyType(
            json.loads(self._compiled_model_fold_report_canonical.decode("ascii"))
        )

    @property
    def report(self) -> Mapping[str, Any]:
        """Return a detached view of immutable catalog/install report bytes."""

        return MappingProxyType(
            json.loads(self._catalog_compile_report_canonical.decode("ascii"))
        )

    def bind_candidate(
        self,
        *,
        compiled_model_fold_report: Mapping[str, Any],
        catalog_compile_report: Mapping[str, Any],
        candidate_state_fingerprint: str,
    ) -> None:
        if not self.active:
            raise Lfm2CpcfCatalogError("Cannot bind an inactive CPCF transaction.")
        candidate = _digest(
            candidate_state_fingerprint,
            "candidate_state_fingerprint",
        )
        if candidate == self.incumbent_state_fingerprint:
            raise Lfm2CpcfCatalogError(
                "CPCF candidate is identical to the compiler-zero target."
            )
        if (
            fingerprint_model_runtime_state(self.context.child_model)
            != self._snapshot.runtime_fingerprint
        ):
            raise Lfm2CpcfCatalogError("CPCF changed target runtime topology.")
        compiler_report = _authenticate_compiled_model_fold_report(
            compiled_model_fold_report,
            v5_backend=self._v5_backend,
            v5_resume_contract=self._v5_resume_contract,
            v5_resume_coordinator=self._v5_resume_coordinator,
        )
        catalog_report = _authenticate_catalog_compile_report(
            catalog_compile_report,
            compiled_model_fold_result_sha256=compiler_report["result_sha256"],
        )
        receipt = catalog_report.get("parameter_path_write_receipt")
        authenticated_paths = _authenticated_compiler_write_paths(catalog_report)
        observed_receipt = _parameter_path_write_receipt(
            self.context.child_model,
            self._snapshot,
            expected_unique_parameters=(
                self.context.child_graph.unique_parameter_count
            ),
            candidate_state_fingerprint=candidate,
            authenticated_compiler_write_paths=authenticated_paths,
        )
        if not isinstance(receipt, Mapping) or dict(receipt) != dict(observed_receipt):
            raise Lfm2CpcfCatalogError(
                "CPCF report lacks the exact independent parameter-path write receipt."
            )
        self._compiled_model_fold_report_canonical = _canonical_json(
            dict(compiler_report)
        )
        self._catalog_compile_report_canonical = _canonical_json(dict(catalog_report))
        self._catalog_compile_report_sha256 = catalog_report["report_sha256"]
        self.candidate_state_fingerprint = candidate

    def verify_parameter_path_write_receipt(self) -> Mapping[str, Any]:
        receipt = self.report.get("parameter_path_write_receipt")
        authenticated_paths = _authenticated_compiler_write_paths(self.report)
        observed = _parameter_path_write_receipt(
            self.context.child_model,
            self._snapshot,
            expected_unique_parameters=(
                self.context.child_graph.unique_parameter_count
            ),
            candidate_state_fingerprint=self.candidate_state_fingerprint,
            authenticated_compiler_write_paths=authenticated_paths,
        )
        if not isinstance(receipt, Mapping) or dict(receipt) != dict(observed):
            raise Lfm2CpcfCatalogError(
                "CPCF live target no longer matches its parameter-path write receipt."
            )
        return observed

    def commit(self) -> Mapping[str, Any]:
        if not self.active:
            raise Lfm2CpcfCatalogError("CPCF transaction is inactive.")
        current = self.context.child_state_fingerprint
        if (
            current != self.candidate_state_fingerprint
            or fingerprint_model_runtime_state(self.context.child_model)
            != self._snapshot.runtime_fingerprint
        ):
            raise Lfm2CpcfCatalogError("CPCF candidate changed before commit.")
        self.verify_parameter_path_write_receipt()
        compiler_report = _authenticate_compiled_model_fold_report(
            self.compiled_model_fold_report,
            v5_backend=self._v5_backend,
            v5_resume_contract=self._v5_resume_contract,
            v5_resume_coordinator=self._v5_resume_coordinator,
        )
        catalog_report = _authenticate_catalog_compile_report(
            self.report,
            compiled_model_fold_result_sha256=compiler_report["result_sha256"],
            expected_report_sha256=self._catalog_compile_report_sha256,
        )
        if catalog_report.get("complete_target_parameter_overwrite") is not True:
            raise Lfm2CpcfCatalogError(
                "CPCF catalog compile report lacks complete installation invariants."
            )
        self.active = False
        return {
            "schema": LFM2_CPCF_CATALOG_SCHEMA,
            "compiled_model_fold": deepcopy(dict(compiler_report)),
            "catalog_compile_report": deepcopy(dict(catalog_report)),
            "action": "commit",
            "accepted": True,
            "exact": True,
            "candidate_state_fingerprint": (self.candidate_state_fingerprint),
            "final_state_fingerprint": current,
            "parameter_topology_preserved": True,
            "zero_gradient": True,
        }

    def rollback(self) -> Mapping[str, Any]:
        if not self.active:
            raise Lfm2CpcfCatalogError("CPCF transaction is inactive.")
        _restore_model(self.context.child_model, self._snapshot)
        final = self.context.child_state_fingerprint
        self.active = False
        return {
            "schema": LFM2_CPCF_CATALOG_SCHEMA,
            "compiled_model_fold": deepcopy(dict(self.compiled_model_fold_report)),
            "catalog_compile_report": deepcopy(dict(self.report)),
            "action": "rollback",
            "accepted": False,
            "exact": True,
            "rollback_exact": True,
            "candidate_state_fingerprint": (self.candidate_state_fingerprint),
            "final_state_fingerprint": final,
            "parameter_topology_preserved": True,
            "zero_gradient": True,
        }


class Lfm2CpcfModelOperator:
    """One full-source/full-target CPCF operator selected by the catalog."""

    method_id = DENSE_BUDGET_MODEL_METHOD

    def __init__(
        self,
        selected_target: Lfm2DenseBudgetedSelectedTarget,
        evidence: Lfm2CpcfSealedEvidence | Lfm2CpcfV5SelectiveSealedEvidence,
    ) -> None:
        if not isinstance(
            selected_target,
            Lfm2DenseBudgetedSelectedTarget,
        ):
            raise Lfm2CpcfCatalogError(
                "selected_target must be Lfm2DenseBudgetedSelectedTarget."
            )
        if not isinstance(
            evidence,
            (Lfm2CpcfSealedEvidence, Lfm2CpcfV5SelectiveSealedEvidence),
        ):
            raise Lfm2CpcfCatalogError(
                "evidence must be sealed legacy or selective-v5 CPCF evidence."
            )
        self.selected_target = selected_target
        self.evidence = evidence.verify()
        self.resource_fingerprint = evidence.bundle_sha256

    def _validate_action(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
    ) -> tuple[str, Any, ModelCircuitGraph]:
        selected = self.selected_target
        if (
            action.method_id != self.method_id
            or len(action.sources) != 1
            or action.compression_delta <= 0
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF requires one positive-compaction whole-model action."
            )
        source = action.sources[0]
        if source.parent_name != selected.source_parent:
            raise Lfm2CpcfCatalogError(
                "CPCF action source differs from target selection."
            )
        parent_graph = context.parent_graphs[source.parent_name]
        if (
            parent_graph.fingerprint != selected.source_graph_fingerprint
            or parent_graph.tokenizer_fingerprint
            != selected.parent_tokenizer_semantics_sha256
            or set(source.node_ids) != {node.node_id for node in parent_graph.nodes}
            or set(action.target_node_ids)
            != {node.node_id for node in context.child_graph.nodes}
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF action must own the complete authenticated parent and "
                "target circuits."
            )
        expected_paths = {
            name for name, _parameter in _named_parameters(context.child_model)
        }
        if set(action.target_parameter_paths) != expected_paths:
            raise Lfm2CpcfCatalogError(
                "CPCF action does not own every target parameter path."
            )
        configuration = dict(action.configuration)
        expected_configuration = {
            "catalog_id": selected.catalog_id,
            "source_depth": len(parent_graph.layer_indices),
            "target_depth": selected.target_depth,
            "source_spans": encode_lfm2_source_spans(selected.ordered_source_spans),
            "target_layer_types": encode_lfm2_layer_types(selected.layer_types),
            "target_hidden_size": selected.hidden_size,
            "target_intermediate_size": selected.intermediate_size,
            "target_query_heads": selected.query_heads,
            "target_key_value_heads": selected.key_value_heads,
            "target_head_dim": selected.head_dim,
            "tokenizer_bridge_required": True,
            "tokenizer_bridge_artifact_sha256": (
                selected.tokenizer_bridge_artifact_sha256
            ),
            "tokenizer_transport_profile_sha256": (
                selected.tokenizer_transport_profile_sha256
            ),
            "parent_tokenizer_semantics_sha256": (
                selected.parent_tokenizer_semantics_sha256
            ),
            "child_tokenizer_semantics_sha256": (
                selected.child_tokenizer_semantics_sha256
            ),
        }
        if any(
            configuration.get(name) != value
            for name, value in expected_configuration.items()
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF action topology or tokenizer bridge differs from the "
                "selected target."
            )
        source_depth = len(parent_graph.layer_indices)
        if (
            selected.target_depth != source_depth
            or selected.ordered_source_spans
            != tuple((index,) for index in range(source_depth))
        ):
            raise Lfm2CpcfCatalogError(
                "Current CPCF model compilation is width/GQA/FFN compaction "
                "at equal depth; depth composition remains fail-closed."
            )
        parent_model = context.parent_models[source.parent_name]
        if fingerprint_model_state(parent_model) != selected.source_state_fingerprint:
            raise Lfm2CpcfCatalogError(
                "CPCF source checkpoint changed after target selection."
            )
        validate_lfm2_dense_budgeted_child(
            selected,
            context.child_model,
            context.child_graph,
        )
        return source.parent_name, parent_model, parent_graph

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
    ) -> FoldTransaction:
        source_name, parent_model, _parent_graph = self._validate_action(
            action,
            context,
        )
        if (
            evidence.payload is not self.evidence
            or evidence.fingerprint != self.evidence.bundle_sha256
            or evidence.root_fingerprint != self.evidence.cache_key
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF fold evidence lost its exact catalog bundle binding."
            )
        sealed = self.evidence.verify_parent_tensor_integrity()
        if (
            context.child_state_fingerprint
            != sealed.blank_target_report["target_blank_state_fingerprint"]
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF target no longer matches its compiler blank receipt."
            )
        _verify_source_tied_table(
            parent_model,
            sealed.source_tied_table,
            sealed.trace_adapter,
        )
        parent_before = fingerprint_model_state(parent_model)
        parent_runtime_before = fingerprint_model_runtime_state(parent_model)
        parent_module_modes = tuple(
            (module, bool(module.training)) for module in parent_model.modules()
        )
        live_snapshot = _capture_model(context.child_model)
        resume_sidecar_objects: dict[str, Any] = {}
        if isinstance(sealed, Lfm2CpcfV5SelectiveSealedEvidence):
            _verify_v5_resume_live_context(
                sealed=sealed,
                parent_model=parent_model,
                live_model=context.child_model,
                selected_target=self.selected_target,
            )
            resume_sidecar_objects = {
                "live_parent_model": parent_model,
                "selective_backend": sealed.backend,
                "selective_cache": sealed.backend.cache,
                "resume_coordinator": sealed.resume_coordinator,
                "resume_state": sealed.resume_state,
                "resume_store": (
                    None
                    if sealed.resume_coordinator is None
                    else sealed.resume_coordinator.store
                ),
            }
            if _model_resume_sidecar_paths(
                context.child_model,
                forbidden_objects=resume_sidecar_objects,
            ):
                raise Lfm2CpcfCatalogError(
                    "Live V5 blank already carries a parent, cache, or resume "
                    "resource sidecar."
                )
        transaction = Lfm2CpcfModelTransaction(
            context,
            live_snapshot,
            v5_backend=(
                sealed.backend
                if isinstance(sealed, Lfm2CpcfV5SelectiveSealedEvidence)
                else None
            ),
            v5_resume_contract=(
                sealed.resume_contract
                if isinstance(sealed, Lfm2CpcfV5SelectiveSealedEvidence)
                else None
            ),
            v5_resume_coordinator=(
                sealed.resume_coordinator
                if isinstance(sealed, Lfm2CpcfV5SelectiveSealedEvidence)
                else None
            ),
        )
        selective_provider: Lfm2CpcfSelectiveEvidenceProvider | None = None
        try:
            staging_model, staging_snapshot = _make_disposable_staging_blank(
                context.child_model,
                live_snapshot,
            )
            validate_lfm2_dense_budgeted_child(
                self.selected_target,
                staging_model,
                context.child_graph,
            )
            guard = _CompileSealedStorageWriteGuard(
                _compile_protected_tensor_bindings(
                    phases=(
                        sealed.phases
                        if isinstance(sealed, Lfm2CpcfSealedEvidence)
                        else ()
                    ),
                    source_tied_table=sealed.source_tied_table,
                    parent_model=parent_model,
                    live_child_model=context.child_model,
                    staging_child_model=staging_model,
                ),
                writable_bindings=_named_parameters(staging_model),
            )
            compile_error: BaseException | None = None
            result: Any = None
            serialized_compiler_report: dict[str, Any] | None = None
            gpu_execution_binding: Mapping[str, Any] | None = None
            try:
                if isinstance(
                    sealed,
                    Lfm2CpcfV5SelectiveSealedEvidence,
                ):
                    selective_provider = sealed.open_provider()
                with guard.mode:
                    if isinstance(sealed, Lfm2CpcfSealedEvidence):
                        result = compile_lfm2_cpcf_model_fold(
                            parent_model,
                            staging_model,
                            phases=sealed.phases,
                            source_tied_table=sealed.source_tied_table,
                            blank_target_report=sealed.blank_target_report,
                            planned_target_unique_parameters=(
                                self.selected_target.exact_target_unique_parameters
                            ),
                            policy=sealed.policy,
                        )
                    else:
                        if selective_provider is None:
                            raise Lfm2CpcfCatalogError(
                                "V5 selective provider was not opened."
                            )
                        backend = sealed.backend
                        compiler = backend.authenticated_compiler()
                        result = compiler(
                            parent_model,
                            staging_model,
                            trace_adapter=sealed.trace_adapter,
                            evidence_provider=selective_provider,
                            source_tied_table=sealed.source_tied_table,
                            blank_target_report=sealed.blank_target_report,
                            planned_target_unique_parameters=(
                                self.selected_target.exact_target_unique_parameters
                            ),
                            runtime_policy=backend.runtime_policy,
                            candidate_replay_evaluator=(
                                backend.candidate_replay_evaluator
                            ),
                            candidate_selector=backend.candidate_selector,
                            numerical_frontier_adapter=(
                                backend.numerical_frontier_adapter
                            ),
                            resume_state=sealed.resume_state,
                            resume_coordinator=sealed.resume_coordinator,
                        )
                    # Result serialization remains compiler-controlled code and
                    # therefore stays inside the same dispatch firewall.
                    serialized_compiler_report = dict(result.as_dict())
            except BaseException as exc:
                compile_error = exc
            finally:
                if selective_provider is not None:
                    selective_provider.close()
            try:
                current_guard_bindings = _compile_protected_tensor_bindings(
                    phases=(
                        sealed.phases
                        if isinstance(
                            sealed,
                            Lfm2CpcfSealedEvidence,
                        )
                        else ()
                    ),
                    source_tied_table=sealed.source_tied_table,
                    parent_model=parent_model,
                    live_child_model=context.child_model,
                    staging_child_model=staging_model,
                )
                guard_report = guard.assert_clean(
                    current_guard_bindings,
                    _named_parameters(staging_model),
                    require_complete_writes=(compile_error is None),
                )
            except BaseException as guard_exc:
                if compile_error is not None:
                    raise guard_exc from compile_error
                raise
            if compile_error is not None:
                raise compile_error
            if serialized_compiler_report is None:
                raise Lfm2CpcfCatalogError(
                    "CPCF compiler returned no authenticated report."
                )
            if isinstance(
                sealed,
                Lfm2CpcfV5SelectiveSealedEvidence,
            ):
                resume_sidecar_objects["selective_provider"] = selective_provider
                staging_sidecar_paths = _model_resume_sidecar_paths(
                    staging_model,
                    forbidden_objects=resume_sidecar_objects,
                )
                if staging_sidecar_paths:
                    raise Lfm2CpcfCatalogError(
                        "V5 compiler attached parent, cache, or resume "
                        "resource sidecars to its disposable staging blank."
                    )
            compiler_report = _authenticate_compiled_model_fold_report(
                serialized_compiler_report,
                expected_result_sha256=getattr(
                    result,
                    "result_sha256",
                    None,
                ),
                v5_backend=(
                    sealed.backend
                    if isinstance(
                        sealed,
                        Lfm2CpcfV5SelectiveSealedEvidence,
                    )
                    else None
                ),
                v5_resume_contract=(
                    sealed.resume_contract
                    if isinstance(
                        sealed,
                        Lfm2CpcfV5SelectiveSealedEvidence,
                    )
                    else None
                ),
                v5_resume_coordinator=(
                    sealed.resume_coordinator
                    if isinstance(
                        sealed,
                        Lfm2CpcfV5SelectiveSealedEvidence,
                    )
                    else None
                ),
            )
            if isinstance(
                sealed,
                Lfm2CpcfV5SelectiveSealedEvidence,
            ):
                _authenticate_v5_compiler_result(
                    result,
                    compiler_report,
                    sealed.backend,
                )
                gpu_execution_binding = _authenticate_v5_gpu_execution_result(
                    result,
                    sealed.backend,
                )
            _assert_model_matches_snapshot_topology(
                staging_model,
                staging_snapshot,
                name="staging candidate",
            )
            validate_lfm2_dense_budgeted_child(
                self.selected_target,
                staging_model,
                context.child_graph,
            )
            staging_candidate = fingerprint_model_state(staging_model)
            if isinstance(
                sealed,
                Lfm2CpcfV5SelectiveSealedEvidence,
            ):
                reviewed_candidate = fingerprint_cpcf_candidate_state(staging_model)
                if reviewed_candidate != sealed.reviewed_candidate_state_fingerprint:
                    raise Lfm2CpcfCatalogError(
                        "V5 deterministic fold produced a child state that "
                        "differs from the reviewed 64-case whole-output "
                        "candidate."
                    )
            staging_write_receipt = _parameter_path_write_receipt(
                staging_model,
                staging_snapshot,
                expected_unique_parameters=(
                    self.selected_target.exact_target_unique_parameters
                ),
                candidate_state_fingerprint=staging_candidate,
                authenticated_compiler_write_paths=(
                    guard_report["authenticated_writable_parameter_paths"]
                ),
            )
            # The dispatch firewall blocks writes before execution.  These
            # canonical post-hashes independently catch non-dispatch storage
            # replacement and preserve the original pre/post content proof.
            sealed.verify_parent_tensor_integrity()
            if fingerprint_model_state(parent_model) != parent_before:
                raise Lfm2CpcfCatalogError(
                    "CPCF detected a non-dispatch/raw-storage parent state "
                    "mutation. The parent resource is fatally invalidated; "
                    "restoration without duplicating the checkpoint cannot "
                    "be authenticated."
                )
            if fingerprint_model_runtime_state(parent_model) != parent_runtime_before:
                raise Lfm2CpcfCatalogError(
                    "CPCF model fold changed parent runtime topology."
                )
            if (
                fingerprint_model_runtime_state(context.child_model)
                != live_snapshot.runtime_fingerprint
                or fingerprint_model_state(context.child_model)
                != live_snapshot.state_fingerprint
            ):
                raise Lfm2CpcfCatalogError(
                    "CPCF compiler reached or changed the live blank before "
                    "authenticated staged installation."
                )
            (
                candidate,
                write_receipt,
                staged_install_receipt,
            ) = _install_staged_candidate(
                live_model=context.child_model,
                live_snapshot=live_snapshot,
                staging_model=staging_model,
                staging_snapshot=staging_snapshot,
                staging_candidate_state_fingerprint=staging_candidate,
                staging_parameter_receipt=staging_write_receipt,
                authenticated_compiler_write_paths=(
                    guard_report["authenticated_writable_parameter_paths"]
                ),
                expected_unique_parameters=(
                    self.selected_target.exact_target_unique_parameters
                ),
            )
            if isinstance(
                sealed,
                Lfm2CpcfV5SelectiveSealedEvidence,
            ):
                live_sidecar_paths = _model_resume_sidecar_paths(
                    context.child_model,
                    forbidden_objects=resume_sidecar_objects,
                )
                if live_sidecar_paths:
                    raise Lfm2CpcfCatalogError(
                        "Authenticated V5 installation attached parent, "
                        "cache, or resume resource sidecars to the live child."
                    )
            catalog_compile_report = _seal_catalog_compile_report(
                {
                    "method_id": self.method_id,
                    "source_parent": source_name,
                    "catalog_id": self.selected_target.catalog_id,
                    "evidence_bundle_sha256": sealed.bundle_sha256,
                    "parent_cache_key": sealed.cache_key,
                    "tokenizer_bridge_artifact_sha256": (
                        sealed.trace_adapter.bridge_artifact_sha256
                    ),
                    "compiler_backend": (
                        "selective_v5"
                        if isinstance(
                            sealed,
                            Lfm2CpcfV5SelectiveSealedEvidence,
                        )
                        else "legacy_v4"
                    ),
                    "compiler_backend_fingerprint": (
                        sealed.backend.backend_fingerprint
                        if isinstance(
                            sealed,
                            Lfm2CpcfV5SelectiveSealedEvidence,
                        )
                        else sealed.bundle_sha256
                    ),
                    "gpu_execution_binding": (
                        None
                        if gpu_execution_binding is None
                        else dict(gpu_execution_binding)
                    ),
                    "gpu_execution_binding_sha256": (
                        None
                        if gpu_execution_binding is None
                        else gpu_execution_binding["binding_sha256"]
                    ),
                    "parameter_path_write_receipt": dict(write_receipt),
                    "parameter_path_write_receipt_sha256": (
                        write_receipt["receipt_sha256"]
                    ),
                    "staging_parameter_path_write_receipt": dict(staging_write_receipt),
                    "staging_parameter_path_write_receipt_sha256": (
                        staging_write_receipt["receipt_sha256"]
                    ),
                    "staged_install_receipt": dict(staged_install_receipt),
                    "staged_install_receipt_sha256": (
                        staged_install_receipt["receipt_sha256"]
                    ),
                    "compile_sealed_storage_write_guard": dict(guard_report),
                    "compile_sealed_storage_write_guard_sha256": (
                        guard_report["report_sha256"]
                    ),
                    "complete_target_parameter_overwrite": (
                        write_receipt[
                            "all_unique_parameter_storages_authentically_written"
                        ]
                        is True
                        and write_receipt["resulting_path_hashes_verified"] is True
                    ),
                    "seed_erasure_by_construction": True,
                    "parameter_topology_preserved": True,
                    "compiler_received_disposable_staging_blank": True,
                    "compiler_received_live_child": False,
                    "live_install_controlled_copy_only": True,
                    "parent_immutable": True,
                    "parent_traces_regenerated": False,
                    "resume_supported": (
                        isinstance(
                            sealed,
                            Lfm2CpcfV5SelectiveSealedEvidence,
                        )
                    ),
                    "resume_used": (
                        isinstance(
                            sealed,
                            Lfm2CpcfV5SelectiveSealedEvidence,
                        )
                        and sealed.resume_state is not None
                    ),
                    "resume_publication_enabled": (
                        isinstance(
                            sealed,
                            Lfm2CpcfV5SelectiveSealedEvidence,
                        )
                        and sealed.resume_coordinator is not None
                    ),
                    "resume_start_layer_index": (
                        compiler_report.get("resume_start_layer_index")
                        if isinstance(
                            sealed,
                            Lfm2CpcfV5SelectiveSealedEvidence,
                        )
                        else None
                    ),
                    "resume_completed_prefix_layer_indices": (
                        compiler_report.get("resume_completed_prefix_layer_indices")
                        if isinstance(
                            sealed,
                            Lfm2CpcfV5SelectiveSealedEvidence,
                        )
                        else []
                    ),
                    "resume_executed_layer_indices": (
                        compiler_report.get("resume_executed_layer_indices")
                        if isinstance(
                            sealed,
                            Lfm2CpcfV5SelectiveSealedEvidence,
                        )
                        else []
                    ),
                    "resume_admission_sha256": (
                        compiler_report.get("resume_admission_sha256")
                        if isinstance(
                            sealed,
                            Lfm2CpcfV5SelectiveSealedEvidence,
                        )
                        else None
                    ),
                    "resume_frontier_sha256": (
                        compiler_report.get("resume_frontier_sha256")
                        if isinstance(
                            sealed,
                            Lfm2CpcfV5SelectiveSealedEvidence,
                        )
                        else None
                    ),
                    "resume_admitted_prefixes": (
                        compiler_report.get("resume_admitted_prefixes")
                        if isinstance(
                            sealed,
                            Lfm2CpcfV5SelectiveSealedEvidence,
                        )
                        else []
                    ),
                    "resume_checkpoint_references_audit_only": True,
                    "resume_checkpoint_tensors_installed_in_live_child": (False),
                    "resume_parent_or_cache_tensor_sidecars": 0,
                    "resume_sidecars_attached_to_live_child": False,
                    "quantization_used": False,
                    "pruning_used": False,
                    "training_used": False,
                    "zero_gradient": True,
                    "optimizer_steps": 0,
                    "backward_calls": 0,
                },
                compiled_model_fold_result_sha256=compiler_report["result_sha256"],
            )
            transaction.bind_candidate(
                compiled_model_fold_report=compiler_report,
                catalog_compile_report=catalog_compile_report,
                candidate_state_fingerprint=candidate,
            )
            return transaction
        except BaseException as exc:
            for module, training in parent_module_modes:
                module.training = training
            try:
                if (
                    fingerprint_model_runtime_state(parent_model)
                    != parent_runtime_before
                ):
                    raise Lfm2CpcfCatalogError(
                        "CPCF failure changed non-mode parent runtime state; "
                        "the parent resource is fatally invalidated."
                    )
            except BaseException as restoration_exc:
                if transaction.active:
                    transaction.rollback()
                raise restoration_exc from exc
            if transaction.active:
                transaction.rollback()
            raise


class Lfm2CpcfFoldOnlyInitializer:
    """Prove that one non-mutating action owns the complete zero target."""

    def __init__(
        self,
        cache_key: str,
        selected_target: Lfm2DenseBudgetedSelectedTarget,
        blank_state_fingerprint: str,
    ) -> None:
        self.cache_key = _digest(cache_key, "cache_key")
        self.selected_target = selected_target
        self.blank_state_fingerprint = _digest(
            blank_state_fingerprint,
            "blank_state_fingerprint",
        )

    def initialize(
        self,
        plan: DynamicFoldPlan,
        context: FoldExecutionContext,
    ) -> Mapping[str, Any]:
        before = context.child_state_fingerprint
        selected = self.selected_target
        if (
            before != self.blank_state_fingerprint
            or plan.child_state_fingerprint != self.blank_state_fingerprint
            or plan.evidence_fingerprint != self.cache_key
            or plan.exact_target_unique_parameters
            != selected.exact_target_unique_parameters
            or len(plan.actions) != 1
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF plan budget, evidence, action count, or blank changed."
            )
        action = plan.actions[0]
        if (
            action.method_id != DENSE_BUDGET_MODEL_METHOD
            or not action.signature.zero_gradient
            or action.compression_delta <= 0
            or set(action.target_node_ids)
            != {node.node_id for node in context.child_graph.nodes}
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF target must compile as one positive-compaction, "
                "zero-gradient, complete-model action."
            )
        if len(context.parent_graphs) != 1 or dict(plan.parent_state_fingerprints) != {
            selected.source_parent: selected.source_state_fingerprint
        }:
            raise Lfm2CpcfCatalogError(
                "CPCF execution parent differs from target selection."
            )
        parent_graph = context.parent_graphs[selected.source_parent]
        if (
            parent_graph.fingerprint != selected.source_graph_fingerprint
            or len(parent_graph.layer_indices) != selected.target_depth
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF source graph differs from its selected equal-depth lane."
            )
        validate_lfm2_dense_budgeted_child(
            selected,
            context.child_model,
            context.child_graph,
        )
        if context.child_state_fingerprint != before:
            raise Lfm2CpcfCatalogError("CPCF initializer mutated the target.")
        return {
            "kind": "lfm2_cpcf_fold_only",
            "child_state_fingerprint": before,
            "complete_target_coverage": True,
            "parent_cache_fingerprint": self.cache_key,
            "parent_traces_regenerated": False,
            "zero_gradient": True,
            "optimizer_steps": 0,
            "backward_calls": 0,
        }


class Lfm2CpcfAdmissionEvaluator:
    """Local structural admission; external intelligence remains mandatory."""

    def evaluate(self, context: FoldEvaluationContext) -> FoldDecision:
        transaction = context.transaction
        if not isinstance(transaction, Lfm2CpcfModelTransaction):
            return FoldDecision.reject(
                "CPCF action lacks its typed whole-model transaction."
            )
        try:
            compiler_report = _authenticate_compiled_model_fold_report(
                transaction.compiled_model_fold_report,
                v5_backend=transaction._v5_backend,
                v5_resume_contract=transaction._v5_resume_contract,
                v5_resume_coordinator=transaction._v5_resume_coordinator,
            )
            catalog_report = _authenticate_catalog_compile_report(
                transaction.report,
                compiled_model_fold_result_sha256=compiler_report["result_sha256"],
            )
            compiler_backend = catalog_report.get("compiler_backend")
            compiler_schema = compiler_report.get("schema")
            if (compiler_backend, compiler_schema) not in {
                ("legacy_v4", LFM2_CPCF_MODEL_FOLD_SCHEMA),
                ("selective_v5", LFM2_CPCF_V5_MODEL_FOLD_SCHEMA),
            }:
                raise Lfm2CpcfCatalogError(
                    "Catalog compiler backend and compiler-owned schema are "
                    "not one exact supported pair."
                )
            path_receipt = transaction.verify_parameter_path_write_receipt()
        except Lfm2CpcfCatalogError:
            compiler_report = {}
            catalog_report = {}
            path_receipt = None
        operation_audit = compiler_report.get("operation_audit")
        receipt = compiler_report.get("parameter_receipt")
        audit_zero = isinstance(operation_audit, Mapping) and all(
            operation_audit.get(name) == 0
            for name in (
                "parent_model_forward_calls",
                "target_model_forward_calls",
                "generation_calls",
                "parent_cache_generation_calls",
                "training_steps",
                "backward_calls",
                "optimizer_steps",
                "parameters_pruned",
                "parameters_quantized",
                "fallback_paths",
            )
        )
        receipt_valid = (
            isinstance(receipt, Mapping)
            and receipt.get("parameter_count_compaction_only") is True
            and receipt.get("checkpoint_byte_ratio_used") is False
            and receipt.get("quantization_used") is False
            and receipt.get("pruning_used") is False
            and receipt.get("target_unique_parameters")
            == context.execution_context.child_graph.unique_parameter_count
        )
        path_receipt_valid = (
            isinstance(path_receipt, Mapping)
            and path_receipt.get("all_named_parameter_paths_accounted") is True
            and path_receipt.get("all_unique_parameter_storages_authentically_written")
            is True
            and path_receipt.get("resulting_path_hashes_verified") is True
            and path_receipt.get("tied_aliases_explicitly_accounted") is True
            and path_receipt.get("receipt_sha256")
            == catalog_report.get("parameter_path_write_receipt_sha256")
        )
        metrics: dict[str, JsonScalar] = {
            "local_gate_only": True,
            "model_level_action": True,
            "all_layers_compiled_in_order": bool(
                compiler_report.get("all_layers_compiled_in_order", False)
            ),
            "actual_folded_upstream_replay_used": bool(
                compiler_report.get(
                    "actual_folded_upstream_replay_used",
                    False,
                )
            ),
            "complete_target_parameter_overwrite": bool(
                catalog_report.get(
                    "complete_target_parameter_overwrite",
                    False,
                )
            ),
            "seed_erasure_by_construction": bool(
                catalog_report.get("seed_erasure_by_construction", False)
            ),
            "source_state_preserved": bool(
                compiler_report.get("source_state_preserved", False)
            ),
            "target_parameter_count_preserved": bool(
                compiler_report.get(
                    "target_parameter_count_preserved",
                    False,
                )
            ),
            "operation_audit_zero": audit_zero,
            "parameter_receipt_valid": receipt_valid,
            "parameter_path_write_receipt_valid": path_receipt_valid,
            "quantization_used": bool(catalog_report.get("quantization_used", True)),
            "pruning_used": bool(catalog_report.get("pruning_used", True)),
            "training_used": bool(catalog_report.get("training_used", True)),
            "zero_gradient": bool(catalog_report.get("zero_gradient", False)),
        }
        required = (
            metrics["all_layers_compiled_in_order"],
            metrics["actual_folded_upstream_replay_used"],
            metrics["complete_target_parameter_overwrite"],
            metrics["seed_erasure_by_construction"],
            metrics["source_state_preserved"],
            metrics["target_parameter_count_preserved"],
            metrics["operation_audit_zero"],
            metrics["parameter_receipt_valid"],
            metrics["parameter_path_write_receipt_valid"],
            not metrics["quantization_used"],
            not metrics["pruning_used"],
            not metrics["training_used"],
            metrics["zero_gradient"],
        )
        if not all(required):
            return FoldDecision.reject(
                "CPCF model transaction failed a fold-only structural invariant.",
                metrics,
            )
        return FoldDecision.accept(
            "CPCF filled the compiler blank by sealed parent-derived circuit "
            "folding; the independent external intelligence gate remains "
            "mandatory.",
            metrics,
        )


class Lfm2CpcfFoldStackAdapter:
    """Higher-priority LFM2 catalog lane for bridged CPCF evidence."""

    adapter_id = LFM2_CPCF_STACK_ADAPTER_ID
    version = 1
    priority = 200
    target_families = ("lfm2",)
    target_modalities = ("text",)
    method_ids = (DENSE_BUDGET_MODEL_METHOD,)

    def __init__(
        self,
        resources: Lfm2CpcfStackResources | Lfm2CpcfV5SelectiveStackResources,
    ) -> None:
        if not isinstance(
            resources,
            (Lfm2CpcfStackResources, Lfm2CpcfV5SelectiveStackResources),
        ):
            raise Lfm2CpcfCatalogError(
                "resources must be legacy or selective-v5 CPCF resources."
            )
        self.resources = resources
        self.resource_fingerprint = _digest(
            resources.resource_fingerprint,
            "resources.resource_fingerprint",
        )

    def _selected_target(
        self,
        target: DynamicTargetSynthesisResult,
    ) -> Lfm2DenseBudgetedSelectedTarget:
        if (
            not isinstance(target, DynamicTargetSynthesisResult)
            or target.adapter_id != Lfm2TargetSynthesisAdapter.adapter_id
            or target.adapter_version != Lfm2TargetSynthesisAdapter.version
            or not isinstance(
                target.adapter_payload,
                Lfm2DenseBudgetedSelectedTarget,
            )
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF stack requires a dense LFM2 catalog target."
            )
        return target.adapter_payload

    def _validate_target(
        self,
        target: DynamicTargetSynthesisResult,
    ) -> tuple[
        Lfm2DenseBudgetedSelectedTarget,
        Mapping[str, Any],
        BehaviorGateBinding,
    ]:
        selected = self._selected_target(target)
        resources = self.resources
        if isinstance(resources, Lfm2CpcfV5SelectiveStackResources):
            return self._validate_v5_target(
                target,
                selected,
                resources,
            )
        binding = _strict_behavior_binding(
            resources.final_evaluator,
            resources.external_behavior_contract,
            name="final_evaluator",
        )
        trace = resources.trace_adapter.verify()
        reviewed = resources.final_evaluator.evidence
        phase_cache_keys = {
            phase.parent_evidence.cache_key for phase in resources.phases
        }
        if (
            target.resource_profile.family != "lfm2"
            or target.resource_profile.modalities != ("text",)
            or target.graph.family != "lfm2"
            or target.graph.model_name != target.request.child_name
            or resources.parent_graph_fingerprint != selected.source_graph_fingerprint
            or reviewed.blank_target_parent_graph_fingerprint
            != selected.source_graph_fingerprint
            or reviewed.live_parent_state_fingerprint
            != selected.source_state_fingerprint
            or reviewed.compiled_model_fold_schema != LFM2_CPCF_MODEL_FOLD_SCHEMA
            or phase_cache_keys != {reviewed.parent_cache_key}
            or reviewed.trace_adapter_sha256 != trace.evidence_sha256
            or not selected.tokenizer_bridge_required
            or trace.bridge_artifact_sha256 != selected.tokenizer_bridge_artifact_sha256
            or trace.transport_profile_fingerprint
            != selected.tokenizer_transport_profile_sha256
            or reviewed.bridge_artifact_sha256
            != selected.tokenizer_bridge_artifact_sha256
            or reviewed.bridge_profile_sha256
            != selected.tokenizer_transport_profile_sha256
            or reviewed.target_tokenizer_semantics_sha256
            != selected.child_tokenizer_semantics_sha256
            or trace.target_tokenizer_semantics_sha256
            != selected.child_tokenizer_semantics_sha256
            or reviewed.candidate_tokenizer_fingerprint
            != trace.target_tokenizer_package_fingerprint
            or trace.target_row_capacity != selected.vocabulary_size
            or target.request.exact_unique_parameters
            != selected.exact_target_unique_parameters
            or target.resource_profile.unique_parameters
            != selected.exact_target_unique_parameters
            or target.graph.unique_parameter_count
            != selected.exact_target_unique_parameters
            or reviewed.exact_target_unique_parameters
            != selected.exact_target_unique_parameters
            or reviewed.target_topology_fingerprint != target.topology_fingerprint
            or reviewed.blank_target_state_fingerprint != target.blank_state_fingerprint
            or dict(target.parent_graph_fingerprints)
            != {selected.source_parent: selected.source_graph_fingerprint}
        ):
            raise Lfm2CpcfCatalogError(
                "Synthesized target differs from CPCF topology, exact budget, "
                "parent, or tokenizer bridge evidence."
            )
        source_depth = sum(len(span) for span in selected.ordered_source_spans)
        if (
            selected.target_depth != source_depth
            or selected.ordered_source_spans
            != tuple((index,) for index in range(source_depth))
            or any(
                len(phase.parent_evidence.layers) != source_depth
                for phase in resources.phases
            )
            or int(resources.source_tied_table.shape[0]) != selected.vocabulary_size
        ):
            raise Lfm2CpcfCatalogError(
                "CPCF evidence is not complete for the selected equal-depth "
                "dense target."
            )
        validate_lfm2_dense_budgeted_child(
            selected,
            target.model,
            target.graph,
        )
        report = _blank_report(target, selected)
        return selected, report, binding

    def _validate_v5_target(
        self,
        target: DynamicTargetSynthesisResult,
        selected: Lfm2DenseBudgetedSelectedTarget,
        resources: Lfm2CpcfV5SelectiveStackResources,
    ) -> tuple[
        Lfm2DenseBudgetedSelectedTarget,
        Mapping[str, Any],
        BehaviorGateBinding,
    ]:
        resources.backend.verify()
        binding = _strict_behavior_binding(
            resources.final_evaluator,
            resources.external_behavior_contract,
            name="final_evaluator",
        )
        trace = resources.trace_adapter.verify()
        reviewed = resources.final_evaluator.evidence
        if (
            target.resource_profile.family != "lfm2"
            or target.resource_profile.modalities != ("text",)
            or target.graph.family != "lfm2"
            or target.graph.model_name != target.request.child_name
            or resources.parent_graph_fingerprint != selected.source_graph_fingerprint
            or reviewed.blank_target_parent_graph_fingerprint
            != selected.source_graph_fingerprint
            or reviewed.live_parent_state_fingerprint
            != selected.source_state_fingerprint
            or reviewed.compiled_model_fold_schema != LFM2_CPCF_V5_MODEL_FOLD_SCHEMA
            or canonical_sha256(dict(reviewed.compiled_model_fold_policy))
            != canonical_sha256(resources.backend.runtime_policy.as_dict())
            or resources.backend.cache.cache_key != reviewed.parent_cache_key
            or reviewed.trace_adapter_sha256 != trace.evidence_sha256
            or not selected.tokenizer_bridge_required
            or trace.bridge_artifact_sha256 != selected.tokenizer_bridge_artifact_sha256
            or trace.transport_profile_fingerprint
            != selected.tokenizer_transport_profile_sha256
            or reviewed.bridge_artifact_sha256
            != selected.tokenizer_bridge_artifact_sha256
            or reviewed.bridge_profile_sha256
            != selected.tokenizer_transport_profile_sha256
            or reviewed.target_tokenizer_semantics_sha256
            != selected.child_tokenizer_semantics_sha256
            or trace.target_tokenizer_semantics_sha256
            != selected.child_tokenizer_semantics_sha256
            or reviewed.candidate_tokenizer_fingerprint
            != trace.target_tokenizer_package_fingerprint
            or trace.target_row_capacity != selected.vocabulary_size
            or target.request.exact_unique_parameters
            != selected.exact_target_unique_parameters
            or target.resource_profile.unique_parameters
            != selected.exact_target_unique_parameters
            or target.graph.unique_parameter_count
            != selected.exact_target_unique_parameters
            or reviewed.exact_target_unique_parameters
            != selected.exact_target_unique_parameters
            or reviewed.target_topology_fingerprint != target.topology_fingerprint
            or reviewed.blank_target_state_fingerprint != target.blank_state_fingerprint
            or dict(target.parent_graph_fingerprints)
            != {selected.source_parent: selected.source_graph_fingerprint}
        ):
            raise Lfm2CpcfCatalogError(
                "Synthesized target differs from selective-v5 topology, "
                "budget, parent, reviewed state/output gate, or tokenizer "
                "bridge evidence."
            )
        source_depth = sum(len(span) for span in selected.ordered_source_spans)
        if (
            selected.target_depth != source_depth
            or selected.target_depth != resources.backend.expected_layer_count
            or selected.ordered_source_spans
            != tuple((index,) for index in range(source_depth))
            or int(resources.source_tied_table.shape[0]) != selected.vocabulary_size
        ):
            raise Lfm2CpcfCatalogError(
                "Selective-v5 evidence is not complete for the selected "
                "equal-depth dense target."
            )
        validate_lfm2_dense_budgeted_child(
            selected,
            target.model,
            target.graph,
        )
        return selected, _blank_report(target, selected), binding

    def supports(self, target: DynamicTargetSynthesisResult) -> bool:
        try:
            self._validate_target(target)
        except Exception:
            # Capability discovery must be fail-closed and must never claim a
            # high-priority CPCF route that build() would reject.  The exact
            # validator is read-only: it reauthenticates the reviewed gate and
            # trace, checks every target/resource binding, validates the live
            # native topology, and verifies compiler-blank provenance.
            return False
        return True

    def build(
        self,
        target: DynamicTargetSynthesisResult,
    ) -> DynamicFoldSystemStack:
        selected, blank_report, binding = self._validate_target(target)
        if isinstance(
            self.resources,
            Lfm2CpcfV5SelectiveStackResources,
        ):
            evidence = Lfm2CpcfV5SelectiveSealedEvidence(
                backend=self.resources.backend,
                trace_adapter=self.resources.trace_adapter,
                source_tied_table=self.resources.source_tied_table,
                blank_target_report=blank_report,
                reviewed_candidate_state_fingerprint=(
                    self.resources.final_evaluator.evidence.candidate_state_fingerprint
                ),
                resume_coordinator=self.resources.resume_coordinator,
                resume_state=self.resources.resume_state,
                resume_admission_sha256=(self.resources.resume_admission_sha256),
                resource_fingerprint=self.resources.resource_fingerprint,
                selected_target_sha256=canonical_sha256(selected.as_dict()),
            )
        else:
            evidence = Lfm2CpcfSealedEvidence(
                phases=self.resources.phases,
                trace_adapter=self.resources.trace_adapter,
                source_tied_table=self.resources.source_tied_table,
                blank_target_report=blank_report,
                policy=self.resources.policy,
            )
        model_loader: Callable[[Path], Any] = self.resources.model_loader
        if isinstance(
            self.resources,
            Lfm2CpcfV5SelectiveStackResources,
        ):
            model_loader = _V5StandaloneModelLoader(
                loader=self.resources.model_loader,
                forbidden_objects=MappingProxyType(
                    {
                        "selective_backend": self.resources.backend,
                        "selective_cache": self.resources.backend.cache,
                        "tokenizer_trace_adapter": self.resources.trace_adapter,
                        "source_tied_table": self.resources.source_tied_table,
                        "resume_coordinator": (self.resources.resume_coordinator),
                        "resume_state": self.resources.resume_state,
                        "resume_store": (
                            None
                            if self.resources.resume_coordinator is None
                            else self.resources.resume_coordinator.store
                        ),
                    }
                ),
                v5_resource_fingerprint=(self.resources.resource_fingerprint),
            )
        return DynamicFoldSystemStack(
            operators=FoldOperatorRegistry(
                (Lfm2CpcfModelOperator(selected, evidence),)
            ),
            evidence_provider=Lfm2CpcfEvidenceProvider(evidence),
            evaluator=Lfm2CpcfAdmissionEvaluator(),
            final_evaluator=self.resources.final_evaluator,
            model_saver=self.resources.model_saver,
            model_loader=model_loader,
            fresh_reload_evaluator=(self.resources.fresh_reload_evaluator),
            saver_id=self.resources.saver_id,
            loader_id=self.resources.loader_id,
            behavior_gate_id=binding.gate_id,
            behavior_gate_version=binding.gate_version,
            behavior_baseline_fingerprint=(binding.baseline_fingerprint),
            behavior_gate_manifest_fingerprint=(binding.manifest_fingerprint),
            behavior_gate_manifest=binding.as_dict(),
            initializer=Lfm2CpcfFoldOnlyInitializer(
                evidence.cache_key,
                selected,
                target.blank_state_fingerprint,
            ),
            snapshot_config=self.resources.snapshot_config,
            evidence_fingerprint=evidence.cache_key,
            publication_recorder=None,
        )


__all__ = [
    "LFM2_CPCF_CATALOG_SCHEMA",
    "LFM2_CPCF_EXTERNAL_GATE_SCHEMA",
    "LFM2_CPCF_PARAMETER_PATH_WRITE_RECEIPT_SCHEMA",
    "LFM2_CPCF_PARENT_TENSOR_SEAL_SCHEMA",
    "LFM2_CPCF_RESOURCE_SCHEMA",
    "LFM2_CPCF_STACK_ADAPTER_ID",
    "LFM2_CPCF_V5_BACKEND_SCHEMA",
    "LFM2_CPCF_V5_EVIDENCE_SCHEMA",
    "LFM2_CPCF_V5_RESOURCE_SCHEMA",
    "Lfm2CpcfAdmissionEvaluator",
    "Lfm2CpcfCatalogError",
    "Lfm2CpcfEvidenceProvider",
    "Lfm2CpcfExternalBehaviorGateContract",
    "Lfm2CpcfFoldOnlyInitializer",
    "Lfm2CpcfFoldStackAdapter",
    "Lfm2CpcfModelOperator",
    "Lfm2CpcfModelTransaction",
    "Lfm2CpcfSealedEvidence",
    "Lfm2CpcfStackResources",
    "Lfm2CpcfV5SelectiveBackend",
    "Lfm2CpcfV5SelectiveSealedEvidence",
    "Lfm2CpcfV5SelectiveStackResources",
]
