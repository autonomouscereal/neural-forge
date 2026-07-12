"""Output-Jacobian audit for the exact Qwen target-native QKVO transaction.

This module does not train or install another residual tranche.  It replays the
existing behavior-distillation loss twice, observes gradients at the live QKVO
projection boundaries, and asks a memory-bounded sketch for right directions
outside the inherited QKVO factor spaces.  Only a four-tensor, target-shaped
right-basis pack can leave the audit.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import operator
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch import Tensor, nn

from neural_forge import hf_output_jacobian_sketch as _sketch_module
from neural_forge import hf_target_native_behavior_distillation as _trainer
from neural_forge.hf_dense_gqa_qkvo_residual_adapter import (
    DenseGQAGeometry,
    DenseGQAQKVOResidualProjection,
)
from neural_forge.hf_output_jacobian_sketch import (
    OUTPUT_JACOBIAN_SKETCH_SCHEMA,
    OutputJacobianBlockIdentity,
    OutputJacobianSketchReport,
    OutputJacobianSketchSpec,
    TwoPassOutputJacobianSketch,
)
from neural_forge.hf_progressive_target_bridge import (
    ExactTargetCompilerTransaction,
)
from neural_forge.hf_target_native_behavior_distillation import (
    BEHAVIOR_FACTOR_PACK_V2_SCHEMA,
    BehaviorDistillationSchedule,
    BehaviorDomainSuite,
)


QKVO_OUTPUT_JACOBIAN_AUDIT_SCHEMA = "neural-forge-qkvo-output-jacobian-audit-v2"
GUIDED_QKVO_RIGHT_BASIS_SCHEMA = "neural-forge-guided-qkvo-right-basis-v2"
_HARD_MAX_BYTES = 256 * 1024 * 1024
_ROLES: tuple[Literal["q", "k", "v", "o"], ...] = ("q", "k", "v", "o")
_PACK_NAMES = {
    "q": "q_orthogonal_right",
    "k": "k_orthogonal_right",
    "v": "v_orthogonal_right",
    "o": "o_orthogonal_right",
}
_WRAPPERS_PER_LAYER = len(_ROLES)
_WRAPPER_COUNT = 30 * _WRAPPERS_PER_LAYER
_FINGERPRINT_PROJECTION_LANES = 16
_FINGERPRINT_VALUES_PER_TENSOR = 6 + _FINGERPRINT_PROJECTION_LANES
_FINGERPRINT_VALUES_PER_WRAPPER = 2 * _FINGERPRINT_VALUES_PER_TENSOR
_FP32_BYTES = 4
_TENSOR_HASH_STAGING_BYTES = 4 * 1024 * 1024
_TOKEN_PATH_SCALAR_TRANSIENT_BYTES = 8
_REPLAY_AUTHENTICATOR_KIND = (
    "seeded-fp32-full-element-order-sensitive-random-projection-v1"
)
_IDENTITY_CHAIN_HEADER = (
    f"{OUTPUT_JACOBIAN_SKETCH_SCHEMA}:ordered-block-identities-v1"
).encode()


class QKVOOutputJacobianAuditError(RuntimeError):
    """Base failure for an uncertifiable QKVO output-Jacobian audit."""


class QKVOOutputJacobianAuditTopologyError(QKVOOutputJacobianAuditError):
    """The live transaction is not the exact 120-site stage-one topology."""


class QKVOOutputJacobianAuditReplayError(QKVOOutputJacobianAuditError):
    """The two behavior-loss/sketch passes did not replay identically."""


class QKVOOutputJacobianAuditLimitError(QKVOOutputJacobianAuditError):
    """The aggregate sketch plan exceeds its hard memory ceiling."""


class QKVOOutputJacobianAuditGateError(QKVOOutputJacobianAuditError):
    """A rejected audit was asked to publish a guided basis pack."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise QKVOOutputJacobianAuditError(
            "Audit evidence is not canonical JSON."
        ) from error


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise QKVOOutputJacobianAuditError(f"{name} must be positive.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise QKVOOutputJacobianAuditError(f"{name} must be positive.") from error
    if result <= 0:
        raise QKVOOutputJacobianAuditError(f"{name} must be positive.")
    return result


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise QKVOOutputJacobianAuditError(f"{name} must be finite and nonnegative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise QKVOOutputJacobianAuditError(
            f"{name} must be finite and nonnegative."
        ) from error
    if not math.isfinite(result) or result < 0.0:
        raise QKVOOutputJacobianAuditError(f"{name} must be finite and nonnegative.")
    return result


def _tensor_sha256(
    value: Tensor,
    *,
    staging_observer: Callable[[int], None] | None = None,
) -> str:
    if not isinstance(value, Tensor):
        raise QKVOOutputJacobianAuditError("Only tensors can enter a tensor hash.")
    tensor = value.detach()
    if tensor.layout != torch.strided or tensor.device.type == "meta":
        raise QKVOOutputJacobianAuditError(
            "Tensor hashing requires a dense non-meta tensor."
        )
    if not tensor.is_contiguous():
        if tensor.device.type != "cpu" or (
            int(tensor.numel() * tensor.element_size()) > _TENSOR_HASH_STAGING_BYTES
        ):
            raise QKVOOutputJacobianAuditError(
                "Large or CUDA tensor hashing requires contiguous storage."
            )
        tensor = tensor.contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    flat = tensor.view(-1)
    elements_per_chunk = max(
        1, _TENSOR_HASH_STAGING_BYTES // max(1, tensor.element_size())
    )
    for start in range(0, int(flat.numel()), elements_per_chunk):
        chunk = flat[start : start + elements_per_chunk]
        staged = chunk.to(device="cpu").contiguous()
        staged_bytes = int(staged.numel() * staged.element_size())
        if staged_bytes > _TENSOR_HASH_STAGING_BYTES:
            raise QKVOOutputJacobianAuditLimitError(
                "Tensor hash staging exceeded its hard chunk bound."
            )
        if staging_observer is not None:
            staging_observer(staged_bytes)
        raw = staged.view(torch.uint8)
        digest.update(memoryview(raw.numpy()))
    return digest.hexdigest()


def _trainer_compatible_tensor_sha256(
    value: Tensor,
    *,
    staging_observer: Callable[[int], None] | None = None,
) -> str:
    """Stream the canonical trainer tensor hash without a full CPU clone."""

    if not isinstance(value, Tensor) or value.layout != torch.strided:
        raise QKVOOutputJacobianAuditError(
            "Trainer-compatible hashing requires a dense tensor."
        )
    tensor = value.detach()
    if not tensor.is_contiguous():
        raise QKVOOutputJacobianAuditError(
            "Trainer-compatible hashing requires contiguous factor storage."
        )
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    flat = tensor.view(-1)
    elements_per_chunk = max(
        1, _TENSOR_HASH_STAGING_BYTES // max(1, tensor.element_size())
    )
    for start in range(0, int(flat.numel()), elements_per_chunk):
        staged = flat[start : start + elements_per_chunk].to(
            device="cpu"
        ).contiguous()
        staged_bytes = int(staged.numel() * staged.element_size())
        if staged_bytes > _TENSOR_HASH_STAGING_BYTES:
            raise QKVOOutputJacobianAuditLimitError(
                "Trainer-compatible hash staging exceeded its hard chunk bound."
            )
        if staging_observer is not None:
            staging_observer(staged_bytes)
        digest.update(memoryview(staged.view(torch.uint8).numpy()))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class QKVOOutputJacobianAuditConfig:
    """Preregistered, exact stage-one audit bounds and aggregate gates."""

    prompt_count: int = 24
    horizon: int = 8
    lanes: tuple[str, str] = ("parent-greedy", "child-greedy")
    rank: int = 8
    sketch_width: int = 16
    seed: int = 0
    min_normal_fraction: float = 0.20
    min_guided_over_random_ratio: float = 1.50
    max_bytes: int = _HARD_MAX_BYTES

    def __post_init__(self) -> None:
        for name in ("prompt_count", "horizon", "rank", "sketch_width", "max_bytes"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if isinstance(self.seed, bool):
            raise QKVOOutputJacobianAuditError("seed must be an integer.")
        try:
            seed = operator.index(self.seed)
        except (TypeError, ValueError, OverflowError) as error:
            raise QKVOOutputJacobianAuditError("seed must be an integer.") from error
        object.__setattr__(self, "seed", seed)
        lanes = tuple(str(value).strip().lower() for value in self.lanes)
        object.__setattr__(self, "lanes", lanes)
        for name in ("min_normal_fraction", "min_guided_over_random_ratio"):
            object.__setattr__(
                self, name, _finite_nonnegative(getattr(self, name), name)
            )
        if self.prompt_count != 24 or self.horizon != 8 or self.rank != 8:
            raise QKVOOutputJacobianAuditError(
                "The exact audit requires 24 prompts, horizon 8, and rank 8."
            )
        if lanes != ("parent-greedy", "child-greedy"):
            raise QKVOOutputJacobianAuditError(
                "The exact audit requires parent-greedy and child-greedy lanes."
            )
        if self.sketch_width < self.rank:
            raise QKVOOutputJacobianAuditError(
                "sketch_width cannot be smaller than rank."
            )
        if self.min_normal_fraction < 0.20:
            raise QKVOOutputJacobianAuditError(
                "The aggregate normal-fraction gate cannot be below 0.20."
            )
        if self.min_guided_over_random_ratio < 1.50:
            raise QKVOOutputJacobianAuditError(
                "The aggregate guided/random gate cannot be below 1.50."
            )
        if self.max_bytes > _HARD_MAX_BYTES:
            raise QKVOOutputJacobianAuditError(
                "max_bytes exceeds the hard 256 MiB audit ceiling."
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["lanes"] = list(self.lanes)
        return payload

    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class QKVOOutputJacobianAuditReport:
    """Canonical report plus the strict runner-facing attestations."""

    payload: Mapping[str, Any]

    @property
    def passed(self) -> bool:
        return bool(self.payload["passed"])

    @property
    def report_sha256(self) -> str:
        return str(self.payload["report_sha256"])

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True, slots=True)
class QKVOOutputJacobianAuditResult:
    """Scaled guided rows and their immutable audit report."""

    guided_right_rows: Mapping[str, Tensor]
    report: QKVOOutputJacobianAuditReport

    def basis_pack_payload(self) -> dict[str, Any]:
        """Return a pack whose authorization bit is true only after both gates."""

        report = self.report.to_dict()
        if report.get("schema") != QKVO_OUTPUT_JACOBIAN_AUDIT_SCHEMA:
            raise QKVOOutputJacobianAuditError(
                "Only an authenticated QKVO audit v2 report can produce a basis pack."
            )
        diagnostic_passed_roles = report.get("diagnostic_passed_roles")
        authorized_roles = report.get("authorized_roles")
        installable_roles = report.get("installable_roles")
        per_role = report.get("per_role")
        derived_diagnostic_roles = (
            []
            if not isinstance(per_role, Mapping) or set(per_role) != set(_ROLES)
            else [
                role
                for role in _ROLES
                if isinstance(per_role[role], Mapping)
                and per_role[role].get("passed") is True
            ]
        )
        if (
            not isinstance(per_role, Mapping)
            or set(per_role) != set(_ROLES)
            or any(
                not isinstance(per_role[role], Mapping)
                or not isinstance(per_role[role].get("passed"), bool)
                for role in _ROLES
            )
            or not isinstance(authorized_roles, list)
            or not isinstance(installable_roles, list)
            or not isinstance(diagnostic_passed_roles, list)
            or diagnostic_passed_roles != derived_diagnostic_roles
            or any(role not in _ROLES for role in authorized_roles)
            or authorized_roles != [role for role in _ROLES if role in authorized_roles]
            or installable_roles != authorized_roles
        ):
            raise QKVOOutputJacobianAuditError(
                "The audit report has malformed ordered role authorization."
            )
        if not isinstance(report.get("passed"), bool):
            raise QKVOOutputJacobianAuditError(
                "The audit decision is inconsistent with role authorization."
            )
        expected_authorized = diagnostic_passed_roles if report["passed"] else []
        if authorized_roles != expected_authorized:
            raise QKVOOutputJacobianAuditError(
                "Global role authorization disagrees with the audit decision."
            )
        site_indices = report.get("site_indices")
        if (
            not isinstance(site_indices, list)
            or not site_indices
            or report.get("site_indices_sha256")
            != _canonical_sha256(site_indices)
        ):
            raise QKVOOutputJacobianAuditError(
                "The audit report has malformed or unauthenticated site indices."
            )
        factor_tensors = {
            _PACK_NAMES[role]: self.guided_right_rows[role]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .clone()
            for role in _ROLES
        }
        factors = [
            _tensor_manifest_row(name, tensor)
            for name, tensor in factor_tensors.items()
        ]
        manifest_sha256 = _canonical_sha256(factors)
        factor_tensor_nbytes = sum(
            int(value.numel() * value.element_size())
            for value in factor_tensors.values()
        )
        memory_accounting = report.get("memory_accounting")
        if (
            not isinstance(memory_accounting, Mapping)
            or memory_accounting.get("basis_payload_cpu_clone_bytes")
            != factor_tensor_nbytes
        ):
            raise QKVOOutputJacobianAuditLimitError(
                "Basis payload clones disagree with the v2 auxiliary-memory plan."
            )
        if manifest_sha256 != report["basis_manifest_sha256"]:
            raise QKVOOutputJacobianAuditError(
                "Live guided rows disagree with the audited basis manifest."
            )
        return {
            "schema": GUIDED_QKVO_RIGHT_BASIS_SCHEMA,
            "factor_state_sha256": report["factor_state_sha256_before"],
            "site_indices": [list(row) for row in site_indices],
            "site_indices_sha256": report["site_indices_sha256"],
            "basis_manifest_sha256": manifest_sha256,
            "tensor_count": len(factor_tensors),
            "total_numel": sum(int(value.numel()) for value in factor_tensors.values()),
            "total_nbytes": sum(
                int(value.numel() * value.element_size())
                for value in factor_tensors.values()
            ),
            "factors": factors,
            "factor_tensors": factor_tensors,
            "diagnostic_passed_roles": list(diagnostic_passed_roles),
            "authorized_roles": list(authorized_roles),
            "installable_roles": list(installable_roles),
            "authorized_for_training": self.report.passed,
            "installer_must_ignore_unauthorized_role_tensors": True,
            "all_four_role_tensors_retained_for_diagnostics": True,
            "contains_model_state": False,
            "contains_parent_state": False,
            "contains_optimizer_state": False,
            "contains_gradients": False,
            "contains_activations": False,
            "contains_hidden_states": False,
        }


def _tensor_manifest_row(name: str, value: Tensor) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "tensor_sha256": _tensor_sha256(value),
        "numel": int(value.numel()),
        "nbytes": int(value.numel() * value.element_size()),
    }


@dataclass(frozen=True, slots=True)
class _ParameterState:
    name: str
    identity: int
    data_pointer: int
    version: int
    requires_grad: bool
    shape: tuple[int, ...]
    dtype: str
    device: str
    sha256: str

    def manifest_row(self) -> dict[str, Any]:
        return asdict(self)


def _parameter_state(
    model: nn.Module,
    *,
    excluded_ids: frozenset[int],
    staging_observer: Callable[[int], None] | None = None,
) -> tuple[_ParameterState, ...]:
    rows: list[_ParameterState] = []
    for name, value in model.named_parameters():
        if id(value) in excluded_ids:
            continue
        rows.append(
            _ParameterState(
                name=name,
                identity=id(value),
                data_pointer=int(value.data_ptr()),
                version=int(value._version),
                requires_grad=bool(value.requires_grad),
                shape=tuple(value.shape),
                dtype=str(value.dtype),
                device=str(value.device),
                sha256=_tensor_sha256(
                    value,
                    staging_observer=staging_observer,
                ),
            )
        )
    return tuple(rows)


def _parameter_state_sha256(values: Sequence[_ParameterState]) -> str:
    return _canonical_sha256([value.manifest_row() for value in values])


def _version_state_sha256(
    entries: Sequence[tuple[str, nn.Parameter]],
) -> str:
    return _canonical_sha256(
        [
            {
                "name": name,
                "identity": id(value),
                "data_pointer": int(value.data_ptr()),
                "version": int(value._version),
            }
            for name, value in entries
        ]
    )


def _qkvo_factor_state_sha256(
    registry: nn.Module,
    *,
    staging_observer: Callable[[int], None] | None = None,
) -> str:
    return _canonical_sha256(
        [
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _tensor_sha256(
                    value,
                    staging_observer=staging_observer,
                ),
            }
            for name, value in registry.named_parameters(recurse=False)
        ]
    )


def _streamed_factor_state_evidence(
    transaction: ExactTargetCompilerTransaction,
    entries: Sequence[tuple[str, nn.Parameter]],
    *,
    staging_observer: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    names = tuple(name for name, _value in entries)
    hashes = tuple(
        _trainer_compatible_tensor_sha256(
            value,
            staging_observer=staging_observer,
        )
        for _name, value in entries
    )
    rows = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "tensor_sha256": digest,
        }
        for (name, value), digest in zip(entries, hashes, strict=True)
    ]
    schema, lexical_rank = _trainer._factor_pack_contract(transaction)
    factor_state_sha256 = _trainer._factor_state_sha256(
        rows,
        schema=schema,
        lexical_rank=lexical_rank,
    )
    return {
        "schema": schema,
        "lexical_rank": lexical_rank,
        "names": names,
        "tensor_sha256s": hashes,
        "factor_state_sha256": factor_state_sha256,
        "parameter_ids": tuple(id(value) for _name, value in entries),
        "parameter_data_ptrs": tuple(
            int(value.data_ptr()) for _name, value in entries
        ),
        "factor_count": len(entries),
        "total_numel": sum(int(value.numel()) for _name, value in entries),
        "total_nbytes": sum(
            int(value.numel() * value.element_size()) for _name, value in entries
        ),
        "largest_factor_nbytes": max(
            int(value.numel() * value.element_size()) for _name, value in entries
        ),
        "capture_mode": "streamed-live-manifest-no-full-cpu-clone",
        "retained_cpu_clone_bytes": 0,
    }


def _validate_inherited_qkvo_factors(registry: nn.Module) -> None:
    """Require every inherited site product to retain its declared rank."""

    rank = _positive_int(getattr(registry, "rank", None), "qkvo rank")
    for role in _ROLES:
        left = getattr(registry, f"{role}_left", None)
        right = getattr(registry, f"{role}_right", None)
        if not isinstance(left, Tensor) or not isinstance(right, Tensor):
            raise QKVOOutputJacobianAuditTopologyError(
                "The inherited QKVO registry is missing a factor tensor."
            )
        if (
            not bool(torch.isfinite(left.detach()).all())
            or not bool(torch.isfinite(right.detach()).all())
            or int(torch.count_nonzero(left.detach()).item()) == 0
            or int(torch.count_nonzero(right.detach()).item()) == 0
        ):
            raise QKVOOutputJacobianAuditTopologyError(
                "All eight inherited QKVO factor tensors must be finite and nonzero."
            )
        if int(left.shape[0]) != 120 or int(right.shape[0]) != 120:
            raise QKVOOutputJacobianAuditTopologyError(
                "Inherited QKVO factors do not cover all 120 sites."
            )
        # Full rank of both small factor sides is equivalent to full rank of
        # their product.  Work on rank-by-rank Gram cores instead of forming a
        # 512/1536-wide dense correction.
        left32 = left.detach().to(torch.float32)
        right32 = right.detach().to(torch.float32)
        left_gram = left32.transpose(1, 2) @ left32
        right_gram = right32 @ right32.transpose(1, 2)
        left_ranks = torch.linalg.matrix_rank(left_gram).detach().cpu()
        right_ranks = torch.linalg.matrix_rank(right_gram).detach().cpu()
        invalid = torch.nonzero(
            (left_ranks != rank) | (right_ranks != rank), as_tuple=False
        ).reshape(-1)
        if int(invalid.numel()) > 0:
            site_row = int(invalid[0].item())
            raise QKVOOutputJacobianAuditTopologyError(
                f"Inherited {role.upper()} site {site_row} product rank is not "
                f"the declared rank {rank}."
            )


def _validate_domain_and_schedule(
    domain: BehaviorDomainSuite,
    schedule: BehaviorDistillationSchedule,
    config: QKVOOutputJacobianAuditConfig,
) -> None:
    if not isinstance(domain, BehaviorDomainSuite):
        raise QKVOOutputJacobianAuditError(
            "basis_domain must be a BehaviorDomainSuite."
        )
    if domain.split_name != "fit" or len(domain.prompts) != config.prompt_count:
        raise QKVOOutputJacobianAuditError(
            "The basis must expose exactly 24 source-candidate prompts as fit only."
        )
    if not isinstance(schedule, BehaviorDistillationSchedule):
        raise QKVOOutputJacobianAuditError(
            "schedule must be a BehaviorDistillationSchedule."
        )
    if schedule.fit_horizon != config.horizon:
        raise QKVOOutputJacobianAuditError(
            "schedule.fit_horizon must match the exact audit horizon."
        )
    if schedule.child_lane_loss_weight <= 0.0:
        raise QKVOOutputJacobianAuditError(
            "The child-greedy lane must have positive loss weight."
        )
    if any(
        len(prompt.token_ids) > schedule.max_prompt_tokens for prompt in domain.prompts
    ):
        raise QKVOOutputJacobianAuditError(
            "A basis prompt exceeds schedule.max_prompt_tokens."
        )


def _validate_exact_topology(
    transaction: ExactTargetCompilerTransaction,
) -> tuple[DenseGQAGeometry, tuple[tuple[int, int], ...]]:
    if not isinstance(transaction, ExactTargetCompilerTransaction):
        raise QKVOOutputJacobianAuditTopologyError(
            "transaction must be ExactTargetCompilerTransaction."
        )
    transaction.assert_live()
    registry = transaction.qkvo_handle.adapter_registry
    geometry = registry.geometry
    expected_geometry = {
        "logical_layers": 30,
        "hidden_size": 1536,
        "query_heads": 16,
        "key_value_heads": 4,
        "head_dim": 128,
        "query_group_width": 512,
    }
    actual_geometry = {
        "logical_layers": geometry.logical_layers,
        "hidden_size": geometry.hidden_size,
        "query_heads": geometry.query_heads,
        "key_value_heads": geometry.key_value_heads,
        "head_dim": geometry.head_dim,
        "query_group_width": geometry.query_group_width,
    }
    if actual_geometry != expected_geometry or registry.site_count != 120:
        raise QKVOOutputJacobianAuditTopologyError(
            "The audit requires exact h=1536,L=30,Hq=16,Hkv=4,d=128 topology."
        )
    sites = tuple(
        (int(layer), int(group))
        for layer, group in registry.site_indices.detach().cpu().tolist()
    )
    expected_sites = tuple((layer, group) for layer in range(30) for group in range(4))
    if sites != expected_sites:
        raise QKVOOutputJacobianAuditTopologyError(
            "The QKVO registry must cover all 120 physical sites in canonical order."
        )
    if transaction.lexical_handle is None:
        raise QKVOOutputJacobianAuditTopologyError(
            "The audit requires an authenticated v2 transaction with lexical factors."
        )
    return geometry, sites


def _sketch_dimensions(role: str, geometry: DenseGQAGeometry) -> tuple[int, int]:
    if role == "q":
        return geometry.query_group_width, geometry.hidden_size
    if role in {"k", "v"}:
        return geometry.head_dim, geometry.hidden_size
    if role == "o":
        return geometry.hidden_size, geometry.query_group_width
    raise QKVOOutputJacobianAuditTopologyError("Unknown QKVO role.")


def _wrapper_dimensions(role: str, geometry: DenseGQAGeometry) -> tuple[int, int]:
    """Return full live-wrapper ``(output, input)`` widths before group slicing."""

    if role == "q":
        return geometry.query_width, geometry.hidden_size
    if role in {"k", "v"}:
        return geometry.key_value_width, geometry.hidden_size
    if role == "o":
        return geometry.hidden_size, geometry.query_width
    raise QKVOOutputJacobianAuditTopologyError("Unknown QKVO role.")


def _sketch_memory_plan(
    *,
    geometry: DenseGQAGeometry,
    site_count: int,
    inherited_rank: int,
    config: QKVOOutputJacobianAuditConfig,
    max_rows: int,
) -> dict[str, Any]:
    """Conservative simultaneous-persistent plus one-operation peak plan."""

    persistent_total = 0
    largest_transient = 0
    for role in _ROLES:
        out_features, in_features = _sketch_dimensions(role, geometry)
        occupied = inherited_rank
        width = config.sketch_width
        rank = config.rank
        range_width = min(width, in_features - occupied)
        occupied_bytes = 4 * occupied * (out_features + in_features)
        pass1_persistent = (
            occupied_bytes + 4 * out_features * width + 8 * in_features * width
        )
        pass2_persistent = (
            occupied_bytes
            + 4 * in_features * (range_width + rank)
            + 4 * out_features * (range_width + rank)
        )
        persistent_total += site_count * max(pass1_persistent, pass2_persistent)
        factor_analysis = (
            8 * inherited_rank * (out_features + in_features)
            + 32 * inherited_rank * inherited_rank
            + max(
                8
                * (
                    out_features * min(out_features, inherited_rank)
                    + inherited_rank * inherited_rank
                    + inherited_rank
                ),
                8
                * (
                    in_features * min(in_features, inherited_rank)
                    + inherited_rank * inherited_rank
                    + inherited_rank
                ),
            )
        )
        pass1_block = 4 * max_rows * (
            3 * out_features + 3 * in_features + 2 * occupied
        ) + 4 * (max_rows * width + in_features * width)
        transition = (
            24 * in_features * width + 16 * in_features * rank + 8 * width * width
        )
        pass2_block = 4 * max_rows * (
            3 * out_features + 3 * in_features + 2 * occupied
        ) + 4 * (
            max_rows * (range_width + rank) + out_features * max(range_width, rank)
        )
        finalize = (
            16 * out_features * range_width
            + 8 * range_width * range_width
            + 8 * range_width
            + 8 * in_features * (range_width + rank)
            + 12 * rank * in_features
        )
        largest_transient = max(
            largest_transient,
            factor_analysis,
            pass1_block,
            transition,
            pass2_block,
            finalize,
        )
    wrapper_transient = _wrapper_hook_transient_bytes(geometry, max_rows)
    maximum_replay_blocks = config.prompt_count * config.horizon * len(config.lanes)
    fingerprint_buffer = (
        maximum_replay_blocks
        * site_count
        * _FINGERPRINT_VALUES_PER_WRAPPER
        * _FP32_BYTES
    )
    fingerprint_host_staging = fingerprint_buffer
    replay_authenticator_coefficients = (
        max_rows
        * max(geometry.hidden_size, geometry.query_width, geometry.key_value_width)
        * _FINGERPRINT_PROJECTION_LANES
        * _FP32_BYTES
    )
    final_guided_pack = sum(
        site_count
        * config.rank
        * _sketch_dimensions(role, geometry)[1]
        * _FP32_BYTES
        for role in _ROLES
    )
    basis_payload_clone = final_guided_pack
    # Token-only replay plans retain no tensors.  Greedy path selection and
    # replay evidence each materialize at most one int64 argmax scalar outside
    # the excluded model-forward activations.
    token_path_scalar_transient = _TOKEN_PATH_SCALAR_TRANSIENT_BYTES
    # The final pack is CPU-resident while the sketches are drained one at a
    # time.  Adding its full size to the all-site persistent upper bound is
    # deliberately conservative and certifies cross-device owned bytes.
    peak = (
        persistent_total
        + largest_transient
        + wrapper_transient
        + fingerprint_buffer
        + fingerprint_host_staging
        + replay_authenticator_coefficients
        + _TENSOR_HASH_STAGING_BYTES
        + final_guided_pack
        + basis_payload_clone
        + token_path_scalar_transient
    )
    return {
        "persistent_upper_bound_bytes": persistent_total,
        "largest_single_operation_bytes": largest_transient,
        "wrapper_hook_transient_bytes": wrapper_transient,
        "maximum_replay_blocks": maximum_replay_blocks,
        "deferred_fingerprint_buffer_upper_bound_bytes": fingerprint_buffer,
        "deferred_fingerprint_host_staging_upper_bound_bytes": (
            fingerprint_host_staging
        ),
        "replay_authenticator_coefficient_bytes": (
            replay_authenticator_coefficients
        ),
        "tensor_hash_cpu_staging_limit_bytes": _TENSOR_HASH_STAGING_BYTES,
        "final_guided_cpu_pack_bytes": final_guided_pack,
        "basis_payload_cpu_clone_bytes": basis_payload_clone,
        "replay_plan_cached_parent_logit_row_count": 0,
        "replay_plan_cached_parent_logit_bytes": 0,
        "replay_plan_cached_child_logit_row_count": 0,
        "replay_plan_cached_child_logit_bytes": 0,
        "replay_plan_cached_tensor_count": 0,
        "replay_plan_cached_tensor_bytes": 0,
        "token_path_scalar_transient_bytes": token_path_scalar_transient,
        "accounted_auxiliary_peak_bytes": peak,
        "max_auxiliary_bytes": config.max_bytes,
        "memory_scope": {
            "bounded": "audit-explicit-auxiliary-tensors-only",
            "includes": [
                "sketch-accumulators-and-bases",
                "deferred-replay-authenticator-buffers",
                "replay-authenticator-coefficients",
                "bounded-tensor-hash-staging",
                "final-guided-cpu-pack",
                "basis-payload-cpu-clone",
                "token-path-planning-and-replay-argmax-scalar",
            ],
            "excludes": [
                "parent-and-child-model-weights",
                "caller-owned-authenticated-continuation-factor-payload",
                "standard-forward-backward-autograd-activations",
                "framework-allocator-reservations-and-overhead",
                "python-object-overhead",
            ],
        },
    }


def _wrapper_hook_transient_bytes(geometry: DenseGQAGeometry, rows: int) -> int:
    """Upper-bound full-wrapper casts and compact-fingerprint temporaries."""

    widths = {
        "q": (geometry.hidden_size, geometry.query_width),
        "k": (geometry.hidden_size, geometry.key_value_width),
        "v": (geometry.hidden_size, geometry.key_value_width),
        "o": (geometry.query_width, geometry.hidden_size),
    }
    largest = 0
    for input_width, output_width in widths.values():
        elements = rows * (input_width + output_width)
        fp32_casts = 4 * elements
        # Finiteness masks plus square/absolute statistic staging are
        # conservatively treated as if both full tensors coexist. Sampled
        # values and indices are tiny but included explicitly.
        fingerprint_staging = 5 * elements + 16 * (8 + 4) * 2
        largest = max(largest, fp32_casts + fingerprint_staging)
    return largest


def _require_memory_within_limit(
    memory: Mapping[str, int],
    config: QKVOOutputJacobianAuditConfig,
    *,
    phase: str,
) -> None:
    peak = int(memory["accounted_auxiliary_peak_bytes"])
    if peak > config.max_bytes:
        raise QKVOOutputJacobianAuditLimitError(
            f"Audit auxiliary tensors exceed the {phase} byte ceiling before backward."
        )


@dataclass(frozen=True, slots=True)
class _ReplayPlan:
    prompt: Any
    parent_tokens: tuple[int, ...]
    child_tokens: tuple[int, ...]
    position_weights: tuple[float, ...]
    child_indices: tuple[int, ...]


def _greedy_token_path(
    model: nn.Module,
    prompt: Sequence[int],
    horizon: int,
) -> tuple[int, ...]:
    """Select a greedy path without retaining full-vocabulary logit rows."""

    prefix = tuple(int(token) for token in prompt)
    selected: list[int] = []
    with torch.no_grad():
        for _step in range(horizon):
            logits = _trainer._last_causal_logits(model, prefix)
            if logits.requires_grad:
                raise QKVOOutputJacobianAuditReplayError(
                    "Token-only greedy planning unexpectedly retained autograd."
                )
            token = int(torch.argmax(logits, dim=-1).item())
            selected.append(token)
            prefix = (*prefix, token)
    return tuple(selected)


def _build_replay_plans(
    transaction: ExactTargetCompilerTransaction,
    parent_model: nn.Module,
    domain: BehaviorDomainSuite,
    schedule: BehaviorDistillationSchedule,
    config: QKVOOutputJacobianAuditConfig,
) -> tuple[_ReplayPlan, ...]:
    plans: list[_ReplayPlan] = []
    for prompt in domain.prompts:
        parent_tokens = _greedy_token_path(
            parent_model, prompt.token_ids, config.horizon
        )
        child_tokens = _greedy_token_path(
            transaction.model, prompt.token_ids, config.horizon
        )
        divergence = _trainer._first_divergence(parent_tokens, child_tokens)
        plans.append(
            _ReplayPlan(
                prompt=prompt,
                parent_tokens=parent_tokens,
                child_tokens=child_tokens,
                position_weights=_trainer._divergence_position_weights(
                    config.horizon, divergence, schedule
                ),
                child_indices=_trainer._child_lane_indices(
                    child_tokens, divergence, schedule
                ),
            )
        )
    return tuple(plans)


@dataclass(frozen=True, slots=True)
class _LossContext:
    lane: str
    prompt_id: str
    prefix_sha256: str
    loss_weight: float
    block_ordinal: int
    prefix_length: int

    def semantic_identity(self, *, layer: int, role: str, group: int, site: int) -> str:
        return _canonical_json(
            {
                "lane": self.lane,
                "prompt_id": self.prompt_id,
                "prefix_token_sha256": self.prefix_sha256,
                "layer": layer,
                "role": role,
                "physical_group": group,
                "site_row": site,
                "loss_weight": self.loss_weight,
                "block_ordinal": self.block_ordinal,
                "prefix_length": self.prefix_length,
            }
        ).decode("ascii")


def _token_sha256(values: Sequence[int]) -> str:
    return _canonical_sha256([int(value) for value in values])


def _write_device_tensor_fingerprint(
    destination: Tensor,
    value: Tensor,
    projection_coefficients: Tensor,
) -> None:
    """Write a full-element probabilistic authenticator without a host sync."""

    if (
        destination.shape != (_FINGERPRINT_VALUES_PER_TENSOR,)
        or destination.dtype != torch.float32
        or value.dtype != torch.float32
        or value.ndim != 2
        or value.device != destination.device
        or projection_coefficients.device != destination.device
        or projection_coefficients.dtype != torch.float32
        or projection_coefficients.ndim != 2
        or int(projection_coefficients.shape[0]) != _FINGERPRINT_PROJECTION_LANES
    ):
        raise QKVOOutputJacobianAuditReplayError(
            "Deferred QKVO fingerprint tensors have incompatible metadata."
        )
    flat = value.reshape(-1)
    count = int(flat.numel())
    if count <= 0 or int(projection_coefficients.shape[1]) < count:
        raise QKVOOutputJacobianAuditReplayError(
            "Deferred QKVO projection authentication is malformed."
        )
    statistics = torch.stack(
        (
            torch.isfinite(flat).all().to(dtype=torch.float32),
            flat.sum(dtype=torch.float32),
            flat.square().sum(dtype=torch.float32),
            flat.abs().sum(dtype=torch.float32),
            flat.min(),
            flat.max(),
        )
    )
    destination[:6].copy_(statistics)
    destination[6:].copy_(
        projection_coefficients[:, :count] @ flat
    )


def _fingerprint_stats_from_cpu(
    values: Tensor,
    *,
    shape: tuple[int, int],
) -> dict[str, Any]:
    if (
        values.device.type != "cpu"
        or values.dtype != torch.float32
        or values.shape != (_FINGERPRINT_VALUES_PER_TENSOR,)
    ):
        raise QKVOOutputJacobianAuditReplayError(
            "Deferred QKVO fingerprint staging is malformed."
        )
    return {
        "shape": list(shape),
        "finite": bool(values[0] == 1.0),
        "sum": float(values[1]),
        "square_sum": float(values[2]),
        "abs_sum": float(values[3]),
        "minimum": float(values[4]),
        "maximum": float(values[5]),
        "full_element_order_sensitive_random_projections": [
            float(item) for item in values[6:].tolist()
        ],
    }


def _fingerprint_sha256_from_cpu(
    values: Tensor,
    *,
    inputs_shape: tuple[int, int],
    delta_shape: tuple[int, int],
) -> str:
    if values.shape != (_FINGERPRINT_VALUES_PER_WRAPPER,):
        raise QKVOOutputJacobianAuditReplayError(
            "A deferred full-wrapper fingerprint has the wrong width."
        )
    if not bool(torch.isfinite(values).all()):
        raise QKVOOutputJacobianAuditReplayError(
            "A hooked QKVO input or output gradient is non-finite."
        )
    inputs = _fingerprint_stats_from_cpu(
        values[:_FINGERPRINT_VALUES_PER_TENSOR], shape=inputs_shape
    )
    delta = _fingerprint_stats_from_cpu(
        values[_FINGERPRINT_VALUES_PER_TENSOR :], shape=delta_shape
    )
    if inputs["finite"] is not True or delta["finite"] is not True:
        raise QKVOOutputJacobianAuditReplayError(
            "A hooked QKVO input or output gradient is non-finite."
        )
    return _canonical_sha256(
        {
            "authenticator_kind": _REPLAY_AUTHENTICATOR_KIND,
            "collision_free": False,
            "binds_every_element": True,
            "inputs": inputs,
            "delta": delta,
        }
    )


def _projection_coefficients(
    max_elements: int,
    *,
    device: torch.device,
    seed: int,
) -> Tensor:
    maximum = _positive_int(max_elements, "projection coefficient width")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    coefficients = torch.empty(
        (_FINGERPRINT_PROJECTION_LANES, maximum),
        dtype=torch.float32,
        device=device,
    )
    coefficients.uniform_(0.5, 1.5, generator=generator)
    return coefficients


def _compact_numeric_fingerprint(inputs: Tensor, delta: Tensor) -> str:
    """Hash a full-element probabilistic replay authenticator."""

    inputs32 = inputs.detach().to(dtype=torch.float32)
    delta32 = delta.detach().to(dtype=torch.float32)
    if inputs32.ndim != 2 or delta32.ndim != 2 or inputs32.device != delta32.device:
        raise QKVOOutputJacobianAuditReplayError(
            "Hooked QKVO fingerprint inputs must be two matrices on one device."
        )
    values = torch.zeros(
        (_FINGERPRINT_VALUES_PER_WRAPPER,),
        dtype=torch.float32,
        device=inputs32.device,
    )
    coefficients = _projection_coefficients(
        max(int(inputs32.numel()), int(delta32.numel())),
        device=inputs32.device,
        seed=0,
    )
    for offset, value in (
        (0, inputs32),
        (_FINGERPRINT_VALUES_PER_TENSOR, delta32),
    ):
        _write_device_tensor_fingerprint(
            values[offset : offset + _FINGERPRINT_VALUES_PER_TENSOR],
            value,
            coefficients,
        )
    return _fingerprint_sha256_from_cpu(
        values.to(device="cpu").contiguous(),
        inputs_shape=tuple(map(int, inputs32.shape)),
        delta_shape=tuple(map(int, delta32.shape)),
    )


def _slice_qkvo_blocks(
    role: str,
    group: int,
    geometry: DenseGQAGeometry,
    flat_inputs: Tensor,
    flat_delta: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return D/X views for one physical group after one full-wrapper cast."""

    if role == "q":
        start = group * geometry.query_group_width
        return flat_delta[:, start : start + geometry.query_group_width], flat_inputs
    if role in {"k", "v"}:
        start = group * geometry.head_dim
        return flat_delta[:, start : start + geometry.head_dim], flat_inputs
    if role == "o":
        start = group * geometry.query_group_width
        return flat_delta, flat_inputs[:, start : start + geometry.query_group_width]
    raise QKVOOutputJacobianAuditTopologyError("Unknown hooked QKVO role.")


class _QKVOHookCollector:
    def __init__(
        self,
        *,
        transaction: ExactTargetCompilerTransaction,
        geometry: DenseGQAGeometry,
        sites: tuple[tuple[int, int], ...],
        sketches: Mapping[tuple[str, int], TwoPassOutputJacobianSketch],
        max_rows: int,
        fingerprint_seed: int,
        hash_staging_observer: Callable[[int], None] | None = None,
    ) -> None:
        self.transaction = transaction
        self.geometry = geometry
        self.sites = sites
        self.sketches = sketches
        self.phase: Literal["pass1", "pass2"] = "pass1"
        self.current: _LossContext | None = None
        self.handles: list[Any] = []
        self.observations: dict[tuple[str, int], list[int]] = defaultdict(
            lambda: [0, 0]
        )
        by_layer: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for row, (layer, group) in enumerate(sites):
            by_layer[layer].append((row, group))
        self.by_layer = dict(by_layer)
        self.wrapper_count = geometry.logical_layers * _WRAPPERS_PER_LAYER
        self._fingerprint_buffer: Tensor | None = None
        self._expected_blocks = 0
        self._fingerprint_seen = bytearray()
        self._fingerprint_contexts: list[_LossContext | None] = []
        self._fingerprint_rows: list[int] = []
        self._fingerprint_callback_count = 0
        self._fingerprint_host_transfer_count = 0
        self._fingerprint_records_hashed = 0
        self._pass_authenticator_sha256: dict[str, str] = {}
        first_attention = transaction.qkvo_handle.attention_modules[0]
        first_projection = getattr(
            first_attention,
            transaction.qkvo_handle.report.projection_names["q"],
            None,
        )
        first_weight = getattr(first_projection, "weight", None)
        if not isinstance(first_weight, Tensor):
            raise QKVOOutputJacobianAuditTopologyError(
                "The QKVO hook collector cannot determine its accumulator device."
            )
        self._fingerprint_device = first_weight.device
        self._maximum_fingerprint_elements = _positive_int(
            max_rows, "maximum fingerprint rows"
        ) * max(
            geometry.hidden_size,
            geometry.query_width,
            geometry.key_value_width,
        )
        projection_seed = int.from_bytes(
            hashlib.sha256(
                (
                    f"{QKVO_OUTPUT_JACOBIAN_AUDIT_SCHEMA}:{fingerprint_seed}:"
                    "full-element-replay-authenticator"
                ).encode("ascii")
            ).digest()[:8],
            "little",
        ) % (2**63)
        self._projection_seed = projection_seed
        self._projection_coefficients: Tensor | None = _projection_coefficients(
            self._maximum_fingerprint_elements,
            device=self._fingerprint_device,
            seed=projection_seed,
        )
        self._projection_coefficient_nbytes = int(
            self._projection_coefficients.numel()
            * self._projection_coefficients.element_size()
        )
        self._projection_coefficient_sha256 = _tensor_sha256(
            self._projection_coefficients,
            staging_observer=hash_staging_observer,
        )

    @property
    def deferred_fingerprint_buffer_nbytes(self) -> int:
        value = self._fingerprint_buffer
        return 0 if value is None else int(value.numel() * value.element_size())

    @property
    def hotpath_instrumentation(self) -> dict[str, Any]:
        return {
            "wrapper_gradient_callback_count": self._fingerprint_callback_count,
            "callback_sync_evidence": "callback_source_audit",
            "pass_boundary_device_to_host_transfer_call_count": (
                self._fingerprint_host_transfer_count
            ),
            "fingerprint_records_hashed": self._fingerprint_records_hashed,
            "fingerprint_values_per_wrapper": _FINGERPRINT_VALUES_PER_WRAPPER,
            "pass_full_element_replay_authenticator_sha256": dict(
                self._pass_authenticator_sha256
            ),
            "replay_authenticator_kind": _REPLAY_AUTHENTICATOR_KIND,
            "replay_authenticator_binds_every_element": True,
            "replay_authenticator_is_collision_free": False,
            "replay_authenticator_projection_lanes": (
                _FINGERPRINT_PROJECTION_LANES
            ),
            "replay_authenticator_seed": self._projection_seed,
            "replay_authenticator_coefficient_sha256": (
                self._projection_coefficient_sha256
            ),
            "replay_authenticator_coefficient_bytes": (
                self._projection_coefficient_nbytes
            ),
        }

    def begin_pass(self, phase: Literal["pass1", "pass2"], block_count: int) -> None:
        if phase not in {"pass1", "pass2"}:
            raise QKVOOutputJacobianAuditReplayError(
                "Deferred fingerprint phase must be pass1 or pass2."
            )
        blocks = _positive_int(block_count, "replay block_count")
        if self._fingerprint_buffer is not None:
            raise QKVOOutputJacobianAuditReplayError(
                "A deferred fingerprint pass is already active."
            )
        self.phase = phase
        self._expected_blocks = blocks
        self._fingerprint_buffer = torch.zeros(
            (blocks, self.wrapper_count, _FINGERPRINT_VALUES_PER_WRAPPER),
            dtype=torch.float32,
            device=self._fingerprint_device,
        )
        slots = blocks * self.wrapper_count
        self._fingerprint_seen = bytearray(slots)
        self._fingerprint_contexts = [None] * blocks
        self._fingerprint_rows = [0] * slots

    def prepare_context(self, context: _LossContext) -> None:
        if self._fingerprint_buffer is None or context.prefix_length <= 0:
            raise QKVOOutputJacobianAuditReplayError(
                "A loss context was prepared outside an active deferred pass."
            )
        if context.block_ordinal < 0 or context.block_ordinal >= self._expected_blocks:
            raise QKVOOutputJacobianAuditReplayError(
                "A loss context ordinal is outside the deferred fingerprint buffer."
            )

    def _record_wrapper_fingerprint(
        self,
        *,
        context: _LossContext,
        layer: int,
        role: str,
        flat_inputs: Tensor,
        flat_delta: Tensor,
    ) -> None:
        buffer = self._fingerprint_buffer
        if buffer is None:
            raise QKVOOutputJacobianAuditReplayError(
                "A QKVO callback occurred outside a deferred fingerprint pass."
            )
        role_index = _ROLES.index(role)  # type: ignore[arg-type]
        wrapper_index = layer * _WRAPPERS_PER_LAYER + role_index
        ordinal = context.block_ordinal
        linear_index = ordinal * self.wrapper_count + wrapper_index
        if self._fingerprint_seen[linear_index]:
            raise QKVOOutputJacobianAuditReplayError(
                "A QKVO wrapper was observed twice in one behavior-loss block."
            )
        expected_out, expected_in = _wrapper_dimensions(role, self.geometry)
        if (
            int(flat_inputs.shape[0]) != context.prefix_length
            or int(flat_delta.shape[0]) != context.prefix_length
            or int(flat_inputs.shape[1]) != expected_in
            or int(flat_delta.shape[1]) != expected_out
        ):
            raise QKVOOutputJacobianAuditReplayError(
                "A QKVO wrapper shape disagrees with its authenticated prefix."
            )
        previous = self._fingerprint_contexts[ordinal]
        if previous is None:
            self._fingerprint_contexts[ordinal] = context
        elif previous != context:
            raise QKVOOutputJacobianAuditReplayError(
                "QKVO wrappers disagreed on one behavior-loss context."
            )
        destination = buffer[ordinal, wrapper_index]
        coefficients = self._projection_coefficients
        if coefficients is None:
            raise QKVOOutputJacobianAuditReplayError(
                "Replay-authenticator coefficients were released before backward."
            )
        for offset, value in (
            (0, flat_inputs),
            (_FINGERPRINT_VALUES_PER_TENSOR, flat_delta),
        ):
            _write_device_tensor_fingerprint(
                destination[offset : offset + _FINGERPRINT_VALUES_PER_TENSOR],
                value,
                coefficients,
            )
        self._fingerprint_seen[linear_index] = 1
        self._fingerprint_rows[linear_index] = context.prefix_length
        self._fingerprint_callback_count += 1

    def finish_pass(self) -> str:
        buffer = self._fingerprint_buffer
        if buffer is None or self._expected_blocks <= 0:
            raise QKVOOutputJacobianAuditReplayError(
                "No deferred fingerprint pass is active."
            )
        if any(value != 1 for value in self._fingerprint_seen) or any(
            context is None for context in self._fingerprint_contexts
        ):
            raise QKVOOutputJacobianAuditReplayError(
                "The deferred fingerprint buffer did not observe every wrapper once."
            )
        phase = self.phase
        staged = buffer.detach().to(device="cpu").contiguous()
        self._fingerprint_host_transfer_count += 1
        try:
            if not bool(torch.isfinite(staged).all()):
                raise QKVOOutputJacobianAuditReplayError(
                    "A hooked QKVO input or output gradient is non-finite."
                )
            numeric_hashes: list[list[str]] = []
            overall = hashlib.sha256(
                b"neural-forge:qkvo-probabilistic-full-element-authenticator-v1"
            )
            overall.update(bytes.fromhex(self._projection_coefficient_sha256))
            for ordinal in range(self._expected_blocks):
                context = self._fingerprint_contexts[ordinal]
                assert context is not None
                overall.update(_canonical_json(asdict(context)))
                block_hashes: list[str] = []
                for wrapper_index in range(self.wrapper_count):
                    role = _ROLES[wrapper_index % _WRAPPERS_PER_LAYER]
                    rows = self._fingerprint_rows[
                        ordinal * self.wrapper_count + wrapper_index
                    ]
                    out_features, in_features = _wrapper_dimensions(
                        role, self.geometry
                    )
                    numeric = _fingerprint_sha256_from_cpu(
                        staged[ordinal, wrapper_index],
                        inputs_shape=(rows, in_features),
                        delta_shape=(rows, out_features),
                    )
                    block_hashes.append(numeric)
                    overall.update(bytes.fromhex(numeric))
                numeric_hashes.append(block_hashes)

            pass_number = 1 if phase == "pass1" else 2
            for role in _ROLES:
                role_index = _ROLES.index(role)
                for site_row, (layer, group) in enumerate(self.sites):
                    wrapper_index = layer * _WRAPPERS_PER_LAYER + role_index
                    chain = hashlib.sha256(_IDENTITY_CHAIN_HEADER)
                    for ordinal, context in enumerate(self._fingerprint_contexts):
                        assert context is not None
                        identity = OutputJacobianBlockIdentity(
                            semantic_identity=context.semantic_identity(
                                layer=layer,
                                role=role,
                                group=group,
                                site=site_row,
                            ),
                            numeric_fingerprint_sha256=(
                                numeric_hashes[ordinal][wrapper_index]
                            ),
                        )
                        chain.update(bytes.fromhex(identity.sha256()))
                    self.sketches[(role, site_row)].bind_prevalidated_identity_chain(
                        pass_number,
                        chain.hexdigest(),
                    )
            digest = overall.hexdigest()
            self._pass_authenticator_sha256[phase] = digest
            self._fingerprint_records_hashed += (
                self._expected_blocks * self.wrapper_count
            )
            return digest
        finally:
            self._fingerprint_buffer = None
            self._expected_blocks = 0
            self._fingerprint_seen = bytearray()
            self._fingerprint_contexts = []
            self._fingerprint_rows = []

    def install(self) -> None:
        handle = self.transaction.qkvo_handle
        for layer in range(self.geometry.logical_layers):
            attention = handle.attention_modules[layer]
            for role in _ROLES:
                wrapper = getattr(attention, handle.report.projection_names[role], None)
                if not isinstance(wrapper, DenseGQAQKVOResidualProjection):
                    raise QKVOOutputJacobianAuditTopologyError(
                        "A required live QKVO projection wrapper is missing."
                    )
                self.handles.append(wrapper.register_forward_hook(self._forward_hook))
        if len(self.handles) != 120:
            self.remove()
            raise QKVOOutputJacobianAuditTopologyError(
                "The audit must register exactly 120 projection hooks."
            )

    def _forward_hook(
        self, module: nn.Module, arguments: tuple[Any, ...], output: Any
    ) -> None:
        context = self.current
        if context is None:
            raise QKVOOutputJacobianAuditReplayError(
                "A hooked child forward occurred outside an authenticated loss block."
            )
        if (
            not isinstance(module, DenseGQAQKVOResidualProjection)
            or len(arguments) != 1
            or not isinstance(arguments[0], Tensor)
            or not isinstance(output, Tensor)
            or not output.requires_grad
        ):
            raise QKVOOutputJacobianAuditTopologyError(
                "A QKVO wrapper did not expose one differentiable tensor boundary."
            )
        inputs = arguments[0].detach()
        layer = module.layer_index
        role = module.role

        def output_gradient_hook(delta: Tensor) -> Tensor:
            # Cast and flatten the full wrapper tensors exactly once.  Physical
            # groups below receive only FP32 views of these two values.
            flat_inputs = inputs.to(dtype=torch.float32).reshape(
                -1, int(inputs.shape[-1])
            )
            flat_delta = (
                delta.detach().to(dtype=torch.float32).reshape(-1, int(delta.shape[-1]))
            )
            self._record_wrapper_fingerprint(
                context=context,
                layer=layer,
                role=role,
                flat_inputs=flat_inputs,
                flat_delta=flat_delta,
            )
            phase_index = 0 if self.phase == "pass1" else 1
            for site_row, group in self.by_layer[layer]:
                d_block, x_block = _slice_qkvo_blocks(
                    role, group, self.geometry, flat_inputs, flat_delta
                )
                sketch = self.sketches[(role, site_row)]
                if self.phase == "pass1":
                    sketch.add_prevalidated_pass1(d_block, x_block)
                else:
                    sketch.add_prevalidated_pass2(d_block, x_block)
                self.observations[(role, site_row)][phase_index] += 1
            return delta

        output.register_hook(output_gradient_hook)

    def remove(self) -> None:
        while self.handles:
            self.handles.pop().remove()
        self._fingerprint_buffer = None
        self._expected_blocks = 0
        self._fingerprint_seen = bytearray()
        self._fingerprint_contexts = []
        self._fingerprint_rows = []
        self._projection_coefficients = None

    @property
    def active_hook_count(self) -> int:
        return len(self.handles)


def _callback_hotpath_source_audit() -> dict[str, Any]:
    """Fail closed if the complete Python callback graph gains a host sync."""

    targets: tuple[tuple[str, Any], ...] = (
        ("collector.forward_hook", _QKVOHookCollector._forward_hook),
        (
            "collector.record_wrapper_fingerprint",
            _QKVOHookCollector._record_wrapper_fingerprint,
        ),
        ("audit.write_device_authenticator", _write_device_tensor_fingerprint),
        ("audit.slice_qkvo_blocks", _slice_qkvo_blocks),
        ("audit.wrapper_dimensions", _wrapper_dimensions),
        (
            "sketch.add_prevalidated_pass1",
            TwoPassOutputJacobianSketch.add_prevalidated_pass1,
        ),
        (
            "sketch.add_prevalidated_pass2",
            TwoPassOutputJacobianSketch.add_prevalidated_pass2,
        ),
        (
            "sketch.prepare_prevalidated_block",
            TwoPassOutputJacobianSketch._prepare_prevalidated_block,
        ),
        ("sketch.reserve_block", TwoPassOutputJacobianSketch._reserve_block),
        ("sketch.inspect_block", TwoPassOutputJacobianSketch._inspect_block),
        (
            "sketch.convert_prevalidated_block",
            TwoPassOutputJacobianSketch._convert_prevalidated_block,
        ),
        ("sketch.accumulate_pass1", TwoPassOutputJacobianSketch._accumulate_pass1),
        ("sketch.accumulate_pass2", TwoPassOutputJacobianSketch._accumulate_pass2),
        ("sketch.normal_blocks", TwoPassOutputJacobianSketch._normal_blocks),
        ("sketch.require_phase", TwoPassOutputJacobianSketch._require_phase),
        ("sketch.require_bytes", TwoPassOutputJacobianSketch._require_bytes),
        (
            "sketch.current_persistent_bytes",
            TwoPassOutputJacobianSketch._current_persistent_bytes,
        ),
        (
            "sketch.block_transient_bytes",
            TwoPassOutputJacobianSketch._block_transient_bytes,
        ),
        (
            "sketch.validate_dense_float_metadata",
            _sketch_module._validate_dense_float_metadata,
        ),
        ("sketch.tensor_bytes", _sketch_module._tensor_bytes),
    )
    banned = (
        ".item(",
        ".cpu(",
        ".numpy(",
        ".tolist(",
        'device="cpu"',
        "device='cpu'",
        "memoryview(",
        "bool(torch.",
    )
    sources = {name: inspect.getsource(value) for name, value in targets}
    violations = {
        name: [token for token in banned if token in source]
        for name, source in sources.items()
        if any(token in source for token in banned)
    }
    if violations:
        raise QKVOOutputJacobianAuditReplayError(
            f"QKVO callback source audit found a host-sync primitive: {violations}."
        )
    hashes = {
        name: hashlib.sha256(source.encode("utf-8")).hexdigest()
        for name, source in sources.items()
    }
    return {
        "kind": "source-audited-complete-python-callback-call-graph-v1",
        "target_count": len(targets),
        "target_source_sha256": hashes,
        "call_graph_sha256": _canonical_sha256(hashes),
        "banned_host_sync_primitive_count": 0,
        "runtime_cuda_profiler_used": False,
        "limitation": (
            "This source audit covers the complete Python callback call graph; "
            "it does not claim to profile synchronization internal to PyTorch kernels."
        ),
    }


def _loss_divisor(
    plans: Sequence[_ReplayPlan], schedule: BehaviorDistillationSchedule
) -> float:
    divisor = 0.0
    for plan in plans:
        divisor += schedule.parent_lane_loss_weight * sum(plan.position_weights)
        divisor += schedule.child_lane_loss_weight * sum(
            plan.position_weights[index] for index in plan.child_indices
        )
    if not math.isfinite(divisor) or divisor <= 0.0:
        raise QKVOOutputJacobianAuditReplayError(
            "The behavior replay produced an invalid loss divisor."
        )
    return divisor


def _replay_block_count(plans: Sequence[_ReplayPlan]) -> int:
    blocks = sum(len(plan.parent_tokens) + len(plan.child_indices) for plan in plans)
    return _positive_int(blocks, "behavior replay block count")


def _replay_loss_pass(
    *,
    transaction: ExactTargetCompilerTransaction,
    parent_model: nn.Module,
    plans: Sequence[_ReplayPlan],
    schedule: BehaviorDistillationSchedule,
    collector: _QKVOHookCollector,
    factors: Sequence[nn.Parameter],
    hash_staging_observer: Callable[[int], None] | None = None,
) -> tuple[str, int, int]:
    divisor = _loss_divisor(plans, schedule)
    evidence: list[dict[str, Any]] = []
    ordinal = 0
    parent_logit_recomputations = 0
    for plan in plans:
        lanes = (
            (
                "parent-greedy",
                plan.parent_tokens,
                tuple(range(len(plan.parent_tokens))),
                schedule.parent_lane_loss_weight,
            ),
            (
                "child-greedy",
                plan.child_tokens,
                plan.child_indices,
                schedule.child_lane_loss_weight,
            ),
        )
        for lane, lane_tokens, indices, lane_weight in lanes:
            active = set(indices)
            prefix = plan.prompt.token_ids
            for lane_step, token in enumerate(lane_tokens):
                if lane_step not in active:
                    prefix = (*prefix, int(token))
                    continue
                with torch.no_grad():
                    parent_logits = _trainer._last_causal_logits(parent_model, prefix)
                if parent_logits.requires_grad:
                    raise QKVOOutputJacobianAuditReplayError(
                        "Recomputed parent logits unexpectedly retained autograd."
                    )
                parent_logit_recomputations += 1
                loss_weight = lane_weight * plan.position_weights[lane_step] / divisor
                context = _LossContext(
                    lane=lane,
                    prompt_id=plan.prompt.prompt_id,
                    prefix_sha256=_token_sha256(prefix),
                    loss_weight=loss_weight,
                    block_ordinal=ordinal,
                    prefix_length=len(prefix),
                )
                prepare_context = getattr(collector, "prepare_context", None)
                if callable(prepare_context):
                    prepare_context(context)
                collector.current = context
                try:
                    child_logits = _trainer._last_causal_logits(
                        transaction.model, prefix
                    )
                finally:
                    collector.current = None
                repetition_tokens: tuple[int, ...] = ()
                if lane == "child-greedy":
                    recent = lane_tokens[
                        max(0, lane_step - schedule.repetition_window) : lane_step
                    ]
                    repetition_tokens = tuple(
                        sorted(
                            {int(value) for value in recent if recent.count(value) >= 2}
                        )
                    )
                loss = (
                    _trainer._training_loss(
                        parent_logits,
                        child_logits,
                        schedule,
                        prefix=prefix,
                        repetition_token_ids=repetition_tokens,
                    )
                    * loss_weight
                )
                if not bool(torch.isfinite(loss.detach()).all()):
                    raise QKVOOutputJacobianAuditReplayError(
                        "Behavior replay loss is non-finite."
                    )
                evidence.append(
                    {
                        "lane": lane,
                        "prompt_id": plan.prompt.prompt_id,
                        "prefix_sha256": context.prefix_sha256,
                        "block_ordinal": ordinal,
                        "loss_weight": loss_weight,
                        "loss": float(loss.detach().cpu().item()),
                        "parent_action": int(torch.argmax(parent_logits).item()),
                        "child_action": int(torch.argmax(child_logits).item()),
                        "parent_logits_source": "recomputed-under-no-grad",
                        "parent_logits_sha256": _tensor_sha256(
                            parent_logits,
                            staging_observer=hash_staging_observer,
                        ),
                        "child_logits_sha256": _tensor_sha256(
                            child_logits,
                            staging_observer=hash_staging_observer,
                        ),
                        "repetition_token_ids": list(repetition_tokens),
                    }
                )
                loss.backward()
                for factor in factors:
                    factor.grad = None
                ordinal += 1
                prefix = (*prefix, int(token))
    if parent_logit_recomputations != ordinal:
        raise QKVOOutputJacobianAuditReplayError(
            "Every replay block must recompute exactly one parent-logit row."
        )
    return _canonical_sha256(evidence), ordinal, parent_logit_recomputations


def _aggregate_reports(
    reports: Sequence[tuple[str, OutputJacobianSketchReport]],
    *,
    config: QKVOOutputJacobianAuditConfig,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    def aggregate(values: Sequence[OutputJacobianSketchReport]) -> dict[str, Any]:
        raw = sum(value.raw_frobenius_sq_estimate for value in values)
        normal = sum(value.normal_frobenius_sq_estimate for value in values)
        guided = sum(value.guided_capture_energy for value in values)
        random = sum(value.random_capture_energy for value in values)
        tiny = torch.finfo(torch.float64).tiny
        normal_fraction = normal / raw if raw > 0.0 else 0.0
        guided_capture = guided / normal if normal > 0.0 else 0.0
        random_capture = random / normal if normal > 0.0 else 0.0
        ratio = guided / max(random, max(normal * 1e-30, tiny))
        metrics = {
            "sketch_count": len(values),
            "raw_energy": raw,
            "normal_energy": normal,
            "guided_energy": guided,
            "random_energy": random,
            "normal_fraction": normal_fraction,
            "guided_capture_fraction": guided_capture,
            "random_capture_fraction": random_capture,
            "guided_over_random_ratio": ratio,
            "normal_fraction_gate_passed": (
                normal_fraction >= config.min_normal_fraction
            ),
            "guided_over_random_gate_passed": (
                ratio >= config.min_guided_over_random_ratio
            ),
        }
        metrics["passed"] = bool(
            metrics["normal_fraction_gate_passed"]
            and metrics["guided_over_random_gate_passed"]
        )
        return metrics

    all_reports = [value for _role, value in reports]
    aggregate_all = aggregate(all_reports)
    per_role = {
        role: aggregate([value for candidate, value in reports if candidate == role])
        for role in _ROLES
    }
    return aggregate_all, per_role


def _diagnostic_passed_roles(
    per_role: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    if set(per_role) != set(_ROLES) or any(
        not isinstance(per_role[role].get("passed"), bool) for role in _ROLES
    ):
        raise QKVOOutputJacobianAuditGateError(
            "Per-role diagnostics do not expose four boolean decisions."
        )
    return [role for role in _ROLES if per_role[role]["passed"] is True]


def _audit_decision(
    aggregate: Mapping[str, Any],
    diagnostic_passed_roles: Sequence[str],
    *,
    replay_authenticator_match: bool,
    mutation_safe: bool,
    gradients_cleared: bool,
    hooks_removed: bool,
    transaction_live: bool,
) -> bool:
    return bool(
        aggregate.get("normal_fraction_gate_passed") is True
        and aggregate.get("guided_over_random_gate_passed") is True
        and bool(diagnostic_passed_roles)
        and replay_authenticator_match
        and mutation_safe
        and gradients_cleared
        and hooks_removed
        and transaction_live
    )


def _source_helper_sha256() -> str:
    names = (
        "_last_causal_logits",
        "_training_loss",
        "_first_divergence",
        "_divergence_position_weights",
        "_child_lane_indices",
    )
    return _canonical_sha256(
        {name: inspect.getsource(getattr(_trainer, name)) for name in names}
    )


def run_qkvo_output_jacobian_audit(
    transaction: ExactTargetCompilerTransaction,
    parent_model: nn.Module,
    basis_domain: BehaviorDomainSuite,
    schedule: BehaviorDistillationSchedule,
    config: QKVOOutputJacobianAuditConfig,
) -> QKVOOutputJacobianAuditResult:
    """Run the deterministic two-pass QKVO audit without an optimizer/update."""

    if not isinstance(config, QKVOOutputJacobianAuditConfig):
        raise QKVOOutputJacobianAuditError(
            "config must be QKVOOutputJacobianAuditConfig."
        )
    geometry, sites = _validate_exact_topology(transaction)
    if not isinstance(parent_model, nn.Module) or parent_model is transaction.model:
        raise QKVOOutputJacobianAuditError(
            "parent_model must be a distinct torch module."
        )
    _validate_domain_and_schedule(basis_domain, schedule, config)
    callback_source_audit = _callback_hotpath_source_audit()

    factor_entries = _trainer._factor_parameters(transaction)
    maximum_hash_staging_bytes = 0

    def observe_hash_staging(staged_bytes: int) -> None:
        nonlocal maximum_hash_staging_bytes
        maximum_hash_staging_bytes = max(maximum_hash_staging_bytes, staged_bytes)

    factor_state_before = _streamed_factor_state_evidence(
        transaction,
        factor_entries,
        staging_observer=observe_hash_staging,
    )
    if factor_state_before["schema"] != BEHAVIOR_FACTOR_PACK_V2_SCHEMA:
        raise QKVOOutputJacobianAuditTopologyError(
            "The live transaction is not an authenticated v2 factor state."
        )
    _validate_inherited_qkvo_factors(transaction.qkvo_handle.adapter_registry)
    factor_ids = frozenset(id(value) for _name, value in factor_entries)
    factor_versions_before = _version_state_sha256(factor_entries)
    inherited_before = _qkvo_factor_state_sha256(
        transaction.qkvo_handle.adapter_registry,
        staging_observer=observe_hash_staging,
    )
    native_before = _parameter_state(
        transaction.model,
        excluded_ids=factor_ids,
        staging_observer=observe_hash_staging,
    )
    native_sha_before = _parameter_state_sha256(native_before)
    for _name, value in factor_entries:
        value.grad = None

    max_rows = (
        max(len(prompt.token_ids) for prompt in basis_domain.prompts)
        + config.horizon
        - 1
    )
    inherited_rank = transaction.qkvo_handle.adapter_registry.rank
    memory = _sketch_memory_plan(
        geometry=geometry,
        site_count=len(sites),
        inherited_rank=inherited_rank,
        config=config,
        max_rows=max_rows,
    )
    _require_memory_within_limit(memory, config, phase="preregistered auxiliary")

    registry = transaction.qkvo_handle.adapter_registry
    sketches: dict[tuple[str, int], TwoPassOutputJacobianSketch] = {}
    for role_index, role in enumerate(_ROLES):
        out_features, in_features = _sketch_dimensions(role, geometry)
        left = getattr(registry, f"{role}_left")
        right = getattr(registry, f"{role}_right")
        for site_row in range(len(sites)):
            sketches[(role, site_row)] = TwoPassOutputJacobianSketch(
                OutputJacobianSketchSpec(
                    out_features=out_features,
                    in_features=in_features,
                    rank=config.rank,
                    sketch_width=config.sketch_width,
                    seed=config.seed + role_index * len(sites) + site_row,
                    max_bytes=config.max_bytes,
                ),
                inherited_left=left[site_row],
                inherited_right=right[site_row],
            )

    phase1_persistent = sum(
        sketch.accounted_persistent_bytes for sketch in sketches.values()
    )
    phase1_transient = max(
        sketch.accounted_batch_transient_bytes(max_rows, 1)
        for sketch in sketches.values()
    )
    fingerprint_buffer_upper = memory[
        "deferred_fingerprint_buffer_upper_bound_bytes"
    ]
    fingerprint_host_staging_upper = memory[
        "deferred_fingerprint_host_staging_upper_bound_bytes"
    ]
    replay_authenticator_coefficient_bytes = memory[
        "replay_authenticator_coefficient_bytes"
    ]
    tensor_hash_staging_limit = memory["tensor_hash_cpu_staging_limit_bytes"]
    token_path_scalar_transient = memory["token_path_scalar_transient_bytes"]
    phase1_actual_peak = (
        phase1_persistent
        + phase1_transient
        + fingerprint_buffer_upper
        + fingerprint_host_staging_upper
        + replay_authenticator_coefficient_bytes
        + tensor_hash_staging_limit
        + token_path_scalar_transient
    )
    wrapper_hook_transient = _wrapper_hook_transient_bytes(geometry, max_rows)
    phase1_actual_peak += wrapper_hook_transient
    memory.update(
        {
            "phase1_actual_persistent_bytes": phase1_persistent,
            "phase1_actual_max_batch_transient_bytes": phase1_transient,
            "phase1_wrapper_hook_transient_bytes": wrapper_hook_transient,
            "phase1_actual_peak_bytes": phase1_actual_peak,
            "manual_plan_covers_phase1": int(
                memory["accounted_auxiliary_peak_bytes"] >= phase1_actual_peak
            ),
        }
    )
    memory["accounted_auxiliary_peak_bytes"] = max(
        memory["accounted_auxiliary_peak_bytes"], phase1_actual_peak
    )
    _require_memory_within_limit(memory, config, phase="actual pass-one")

    collector = _QKVOHookCollector(
        transaction=transaction,
        geometry=geometry,
        sites=sites,
        sketches=sketches,
        max_rows=max_rows,
        fingerprint_seed=config.seed,
        hash_staging_observer=observe_hash_staging,
    )
    parent_modes = tuple(
        (module, bool(module.training)) for module in parent_model.modules()
    )
    pass1_sha = ""
    pass2_sha = ""
    pass1_authenticator_sha = ""
    pass2_authenticator_sha = ""
    replay_blocks = 0
    hooks_registered = 0
    try:
        parent_model.eval()
        with transaction.output_training_scope() as trainable:
            if {id(value) for value in trainable} != factor_ids:
                raise QKVOOutputJacobianAuditTopologyError(
                    "Output-training scope disagrees with the authenticated v2 factors."
                )
            for value in trainable:
                value.grad = None
            plans = _build_replay_plans(
                transaction, parent_model, basis_domain, schedule, config
            )
            expected_replay_blocks = _replay_block_count(plans)
            actual_fingerprint_bytes = (
                expected_replay_blocks
                * collector.wrapper_count
                * _FINGERPRINT_VALUES_PER_WRAPPER
                * _FP32_BYTES
            )
            if actual_fingerprint_bytes > fingerprint_buffer_upper:
                raise QKVOOutputJacobianAuditLimitError(
                    "The actual deferred fingerprint buffer exceeds its preregistered "
                    "upper bound."
                )
            memory["actual_replay_blocks"] = expected_replay_blocks
            memory["actual_deferred_fingerprint_buffer_bytes"] = (
                actual_fingerprint_bytes
            )
            collector.install()
            hooks_registered = len(collector.handles)
            try:
                collector.begin_pass("pass1", expected_replay_blocks)
                pass1_sha, replay_blocks, pass1_parent_logit_recomputations = (
                    _replay_loss_pass(
                        transaction=transaction,
                        parent_model=parent_model,
                        plans=plans,
                        schedule=schedule,
                        collector=collector,
                        factors=trainable,
                        hash_staging_observer=observe_hash_staging,
                    )
                )
                if replay_blocks != expected_replay_blocks:
                    raise QKVOOutputJacobianAuditReplayError(
                        "Behavior replay did not execute its preregistered block count."
                    )
                pass1_authenticator_sha = collector.finish_pass()
                for sketch in sketches.values():
                    sketch.finish_pass1()
                phase2_persistent = sum(
                    sketch.accounted_persistent_bytes for sketch in sketches.values()
                )
                phase2_transient = max(
                    sketch.accounted_batch_transient_bytes(max_rows, 2)
                    for sketch in sketches.values()
                )
                phase2_actual_peak = (
                    phase2_persistent
                    + phase2_transient
                    + wrapper_hook_transient
                    + 2 * actual_fingerprint_bytes
                    + replay_authenticator_coefficient_bytes
                    + tensor_hash_staging_limit
                    + token_path_scalar_transient
                )
                memory.update(
                    {
                        "phase2_actual_persistent_bytes": phase2_persistent,
                        "phase2_actual_max_batch_transient_bytes": phase2_transient,
                        "phase2_wrapper_hook_transient_bytes": wrapper_hook_transient,
                        "phase2_actual_peak_bytes": phase2_actual_peak,
                        "manual_plan_covers_phase2": int(
                            memory["accounted_auxiliary_peak_bytes"] >= phase2_actual_peak
                        ),
                    }
                )
                memory["accounted_auxiliary_peak_bytes"] = max(
                    memory["accounted_auxiliary_peak_bytes"], phase2_actual_peak
                )
                _require_memory_within_limit(memory, config, phase="actual pass-two")
                collector.begin_pass("pass2", expected_replay_blocks)
                pass2_sha, pass2_blocks, pass2_parent_logit_recomputations = (
                    _replay_loss_pass(
                        transaction=transaction,
                        parent_model=parent_model,
                        plans=plans,
                        schedule=schedule,
                        collector=collector,
                        factors=trainable,
                        hash_staging_observer=observe_hash_staging,
                    )
                )
                pass2_authenticator_sha = collector.finish_pass()
                if (
                    pass2_blocks != replay_blocks
                    or pass1_parent_logit_recomputations != replay_blocks
                    or pass2_parent_logit_recomputations != replay_blocks
                    or pass2_sha != pass1_sha
                    or pass2_authenticator_sha != pass1_authenticator_sha
                ):
                    raise QKVOOutputJacobianAuditReplayError(
                        "Behavior loss replay changed between sketch passes."
                    )
            finally:
                collector.remove()
                collector.current = None
                for value in trainable:
                    value.grad = None
    finally:
        collector.remove()
        collector.current = None
        for module, training in parent_modes:
            module.training = training
        for _name, value in factor_entries:
            value.grad = None

    expected_observations = {
        (role, row) for role in _ROLES for row in range(len(sites))
    }
    if set(collector.observations) != expected_observations or any(
        counts != [replay_blocks, replay_blocks]
        for counts in collector.observations.values()
    ):
        raise QKVOOutputJacobianAuditReplayError(
            "Not all 120 sites and four roles were observed identically in both passes."
        )

    hotpath_instrumentation = collector.hotpath_instrumentation
    hotpath_instrumentation["callback_source_audit"] = callback_source_audit
    expected_callback_count = replay_blocks * collector.wrapper_count * 2
    if (
        hotpath_instrumentation["wrapper_gradient_callback_count"]
        != expected_callback_count
        or hotpath_instrumentation["callback_sync_evidence"]
        != "callback_source_audit"
        or callback_source_audit["banned_host_sync_primitive_count"] != 0
        or hotpath_instrumentation[
            "pass_boundary_device_to_host_transfer_call_count"
        ]
        != 2
        or hotpath_instrumentation["fingerprint_records_hashed"]
        != expected_callback_count
        or hotpath_instrumentation["replay_authenticator_coefficient_bytes"]
        != replay_authenticator_coefficient_bytes
        or hotpath_instrumentation["replay_authenticator_binds_every_element"]
        is not True
        or hotpath_instrumentation["replay_authenticator_is_collision_free"]
        is not False
    ):
        raise QKVOOutputJacobianAuditReplayError(
            "Deferred fingerprint hot-path instrumentation is inconsistent."
        )
    memory["actual_replay_authenticator_coefficient_bytes"] = (
        hotpath_instrumentation["replay_authenticator_coefficient_bytes"]
    )

    expected_shapes = {
        "q": (120, 8, 1536),
        "k": (120, 8, 1536),
        "v": (120, 8, 1536),
        "o": (120, 8, 512),
    }
    guided = {
        role: torch.empty(expected_shapes[role], dtype=torch.float32, device="cpu")
        for role in _ROLES
    }
    scale_records: dict[str, list[float]] = {role: [] for role in _ROLES}
    for role in _ROLES:
        inherited_right = (
            getattr(registry, f"{role}_right")
            .detach()
            .to(dtype=torch.float32)
        )
        scales = (
            torch.linalg.vector_norm(inherited_right, dim=2)
            .median(dim=1)
            .values.detach()
            .to(device="cpu", dtype=torch.float32)
        )
        for scale_value in scales:
            scale = float(scale_value)
            if not math.isfinite(scale) or scale <= 0.0:
                raise QKVOOutputJacobianAuditReplayError(
                    "An inherited-right site has no finite positive median row norm."
                )
            scale_records[role].append(scale)

    sketch_reports: dict[tuple[str, int], OutputJacobianSketchReport] = {}
    for role in _ROLES:
        for site_row in range(len(sites)):
            sketch = sketches.pop((role, site_row))
            result = sketch.finalize()
            guided[role][site_row].copy_(result.guided_right_rows.detach())
            guided[role][site_row].mul_(scale_records[role][site_row])
            sketch_reports[(role, site_row)] = result.report
            del result, sketch
    if sketches or collector.sketches:
        raise QKVOOutputJacobianAuditReplayError(
            "Finalized sketches were not drained from accelerator memory."
        )
    report_pairs = [
        (role, sketch_reports[(role, row)])
        for role in _ROLES
        for row in range(len(sites))
    ]
    aggregate, per_role = _aggregate_reports(report_pairs, config=config)

    if any(tuple(guided[role].shape) != expected_shapes[role] for role in _ROLES):
        raise QKVOOutputJacobianAuditTopologyError(
            "Scaled guided rows do not have the exact stage-one pack shapes."
        )
    factor_tensors = {_PACK_NAMES[role]: guided[role] for role in _ROLES}
    basis_manifest = [
        _tensor_manifest_row(name, value) for name, value in factor_tensors.items()
    ]
    basis_manifest_sha256 = _canonical_sha256(basis_manifest)
    basis_total_numel = sum(int(value.numel()) for value in guided.values())
    if basis_total_numel != 4_915_200:
        raise QKVOOutputJacobianAuditTopologyError(
            "The guided QKVO basis must contain exactly 4,915,200 elements."
        )
    basis_total_nbytes = sum(
        int(value.numel() * value.element_size()) for value in guided.values()
    )
    if basis_total_nbytes != memory["final_guided_cpu_pack_bytes"]:
        raise QKVOOutputJacobianAuditLimitError(
            "The drained guided pack disagrees with its auxiliary memory plan."
        )
    memory.update(
        {
            "actual_final_guided_cpu_pack_bytes": basis_total_nbytes,
            "finalized_result_tensors_retained_on_accumulator_device": 0,
            "all_sketches_drained_after_finalize": 1,
        }
    )

    transaction.assert_live()
    factor_state_after = _streamed_factor_state_evidence(
        transaction,
        factor_entries,
        staging_observer=observe_hash_staging,
    )
    factor_versions_after = _version_state_sha256(factor_entries)
    inherited_after = _qkvo_factor_state_sha256(
        registry,
        staging_observer=observe_hash_staging,
    )
    native_after = _parameter_state(
        transaction.model,
        excluded_ids=factor_ids,
        staging_observer=observe_hash_staging,
    )
    native_sha_after = _parameter_state_sha256(native_after)
    if maximum_hash_staging_bytes > _TENSOR_HASH_STAGING_BYTES:
        raise QKVOOutputJacobianAuditLimitError(
            "Observed tensor-hash staging exceeded its hard bound."
        )
    memory.update(
        {
            "maximum_observed_tensor_hash_cpu_staging_bytes": (
                maximum_hash_staging_bytes
            ),
            "factor_state_capture_mode": factor_state_before["capture_mode"],
            "factor_state_total_nbytes": factor_state_before["total_nbytes"],
            "factor_state_retained_cpu_clone_bytes": 0,
            "factor_state_before_after_capture_overlap_bytes": 0,
        }
    )
    gradients_cleared = all(value.grad is None for _name, value in factor_entries)
    probabilistic_replay_authenticator_match = (
        pass1_sha == pass2_sha
        and pass1_authenticator_sha == pass2_authenticator_sha
    )
    mutation_safe = (
        factor_state_after["factor_state_sha256"]
        == factor_state_before["factor_state_sha256"]
        and factor_versions_after == factor_versions_before
        and inherited_after == inherited_before
        and native_sha_after == native_sha_before
        and native_after == native_before
    )
    diagnostic_passed_roles = _diagnostic_passed_roles(per_role)
    passed = _audit_decision(
        aggregate,
        diagnostic_passed_roles,
        replay_authenticator_match=probabilistic_replay_authenticator_match,
        mutation_safe=mutation_safe,
        gradients_cleared=gradients_cleared,
        hooks_removed=collector.active_hook_count == 0,
        transaction_live=transaction.is_live,
    )
    authorized_roles = list(diagnostic_passed_roles) if passed else []
    installable_roles = list(authorized_roles)
    site_indices = [[int(layer), int(group)] for layer, group in sites]
    site_indices_sha256 = _canonical_sha256(site_indices)

    site_reports = []
    for role in _ROLES:
        for row, (layer, group) in enumerate(sites):
            site_reports.append(
                {
                    "role": role,
                    "site_row": row,
                    "layer": layer,
                    "physical_group": group,
                    "inherited_right_median_row_norm": scale_records[role][row],
                    "sketch_report": sketch_reports[(role, row)].to_dict(),
                }
            )
    report_payload: dict[str, Any] = {
        "schema": QKVO_OUTPUT_JACOBIAN_AUDIT_SCHEMA,
        "passed": passed,
        "config": config.to_dict(),
        "config_sha256": config.sha256(),
        "site_count": 120,
        "role_count": 4,
        "sketch_count": 480,
        "site_indices": site_indices,
        "site_indices_sha256": site_indices_sha256,
        "basis_prompt_count": 24,
        "basis_domain_sha256": basis_domain.domain_sha256,
        "basis_artifact_sha256": basis_domain.artifact_sha256,
        "basis_source_role": "source-exposed-candidate",
        "source_exposed_candidate_as_fit": True,
        "new_candidate_accessed": False,
        "sealed_accessed": False,
        "optimizer_created": False,
        "optimizer_used": False,
        "optimizer_steps": 0,
        "hooks_registered": hooks_registered,
        "hooks_removed": collector.active_hook_count == 0,
        "active_hook_count_after": collector.active_hook_count,
        "transaction_live_after": transaction.is_live,
        "rollback_compatible_after": transaction.is_live,
        "native_state_sha256_before": native_sha_before,
        "native_state_sha256_after": native_sha_after,
        "factor_state_sha256_before": factor_state_before["factor_state_sha256"],
        "factor_state_sha256_after": factor_state_after["factor_state_sha256"],
        "factor_state_capture": {
            "mode": factor_state_before["capture_mode"],
            "factor_count": factor_state_before["factor_count"],
            "total_numel": factor_state_before["total_numel"],
            "total_nbytes": factor_state_before["total_nbytes"],
            "largest_factor_nbytes": factor_state_before["largest_factor_nbytes"],
            "retained_cpu_clone_bytes_before_sketch": factor_state_before[
                "retained_cpu_clone_bytes"
            ],
            "retained_cpu_clone_bytes_after_audit": factor_state_after[
                "retained_cpu_clone_bytes"
            ],
            "maximum_observed_hash_staging_bytes": maximum_hash_staging_bytes,
            "hash_staging_limit_bytes": _TENSOR_HASH_STAGING_BYTES,
            "before_after_captures_overlap": False,
        },
        "inherited_qkvo_state_sha256_before": inherited_before,
        "inherited_qkvo_state_sha256_after": inherited_after,
        "factor_parameter_versions_sha256_before": factor_versions_before,
        "factor_parameter_versions_sha256_after": factor_versions_after,
        "factor_gradients_cleared": gradients_cleared,
        "pass1_replay_sha256": pass1_sha,
        "pass2_replay_sha256": pass2_sha,
        "pass1_full_element_replay_authenticator_sha256": (
            pass1_authenticator_sha
        ),
        "pass2_full_element_replay_authenticator_sha256": (
            pass2_authenticator_sha
        ),
        "probabilistic_replay_authenticator_match": (
            probabilistic_replay_authenticator_match
        ),
        "replay_authentication_kind": _REPLAY_AUTHENTICATOR_KIND,
        "replay_authentication_binds_every_element": True,
        "replay_authentication_is_collision_free": False,
        "replay_authentication_projection_lanes": (
            _FINGERPRINT_PROJECTION_LANES
        ),
        "replay_loss_block_count": replay_blocks,
        "replay_plan_evidence": {
            "mode": "token-paths-only-no-cached-logit-tensors",
            "prompt_count": len(plans),
            "cached_parent_logit_row_count": 0,
            "cached_parent_logit_bytes": 0,
            "cached_child_logit_row_count": 0,
            "cached_child_logit_bytes": 0,
            "cached_tensor_count": 0,
            "cached_tensor_bytes": 0,
            "token_path_scalar_transient_bytes": (
                memory["token_path_scalar_transient_bytes"]
            ),
            "parent_logits_recomputed_under_no_grad": True,
            "parent_logits_sha256_in_replay_evidence": True,
            "parent_logit_recomputations_per_pass": replay_blocks,
            "parent_logit_recomputations_total": 2 * replay_blocks,
        },
        "accounted_auxiliary_peak_bytes": memory["accounted_auxiliary_peak_bytes"],
        "max_auxiliary_bytes": config.max_bytes,
        "memory_scope": memory["memory_scope"],
        "memory_accounting": memory,
        "hotpath_instrumentation": hotpath_instrumentation,
        "aggregate": aggregate,
        "per_role": per_role,
        "diagnostic_passed_roles": diagnostic_passed_roles,
        "authorized_roles": authorized_roles,
        "installable_roles": installable_roles,
        "role_authorization_policy": (
            "installers-must-ignore-all-tensors-for-roles-not-listed-in-"
            "authorized_roles"
        ),
        "installer_must_ignore_unauthorized_role_tensors": True,
        "all_four_role_tensors_retained_for_diagnostics": True,
        "basis_manifest": basis_manifest,
        "basis_manifest_sha256": basis_manifest_sha256,
        "basis_total_numel": basis_total_numel,
        "source_trainer_helper_sha256": _source_helper_sha256(),
        "site_reports": site_reports,
        "internal_weight_matching_used": False,
        "activation_targets_used": False,
        "new_tranche_installed": False,
    }
    report_payload["report_sha256"] = _canonical_sha256(report_payload)
    report = QKVOOutputJacobianAuditReport(MappingProxyType(dict(report_payload)))
    return QKVOOutputJacobianAuditResult(
        guided_right_rows=MappingProxyType(guided),
        report=report,
    )


__all__ = [
    "GUIDED_QKVO_RIGHT_BASIS_SCHEMA",
    "QKVO_OUTPUT_JACOBIAN_AUDIT_SCHEMA",
    "QKVOOutputJacobianAuditConfig",
    "QKVOOutputJacobianAuditError",
    "QKVOOutputJacobianAuditGateError",
    "QKVOOutputJacobianAuditLimitError",
    "QKVOOutputJacobianAuditReplayError",
    "QKVOOutputJacobianAuditReport",
    "QKVOOutputJacobianAuditResult",
    "QKVOOutputJacobianAuditTopologyError",
    "run_qkvo_output_jacobian_audit",
]
