"""Model-independent behavior-guided Gate/Up/Down right-basis audit.

This module is deliberately below model replay and adapter installation.  A
caller supplies ordered, role-local ``(output_delta, projection_input)``
blocks.  The core then combines the existing two-pass output-Jacobian sketch
with the existing balanced-Krylov implementation.  It never compares parent
and child weights and it owns no model hook.

The balanced calculation is performed in a bounded, deterministic envelope
of the role's input space.  This is important for the 5,120-wide Down role:
calling the generic balanced implementation directly would construct a dense
``width x width`` projector.  The envelope preserves output-sensitivity and
activation directions while making that full-width allocation unnecessary.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from neural_forge.balanced_krylov import (
    BalancedKrylovError,
    build_balanced_krylov_basis,
)
from neural_forge.hf_output_jacobian_sketch import (
    OutputJacobianBlockIdentity,
    OutputJacobianSketchError,
    OutputJacobianSketchResult,
    OutputJacobianSketchSpec,
    TwoPassOutputJacobianSketch,
)


GUD_OUTPUT_JACOBIAN_AUDIT_SCHEMA = "neural-forge-gud-output-jacobian-balanced-audit-v1"
GUD_CANDIDATE_BASIS_SCHEMA = "neural-forge-gud-candidate-right-bases-v1"
GUD_ROLE_ORDER: tuple[Literal["gate", "up", "down"], ...] = (
    "gate",
    "up",
    "down",
)
HARD_MAX_AUXILIARY_BYTES = 256 * 1024 * 1024
DEFAULT_FINGERPRINT_CHUNK_BYTES = 1024 * 1024
QWEN_STAGE_LOGICAL_LAYERS = 30
QWEN_STAGE_HIDDEN_SIZE = 1536
QWEN_STAGE_INTERMEDIATE_SIZE = 5120

GUDRole = Literal["gate", "up", "down"]
CandidateKind = Literal[
    "output-jacobian",
    "balanced",
    "procrustes-hybrid",
    "random-control",
]
_CANDIDATE_ORDER: tuple[CandidateKind, ...] = (
    "output-jacobian",
    "balanced",
    "procrustes-hybrid",
    "random-control",
)


class GUDOutputJacobianAuditError(RuntimeError):
    """Base failure for an uncertifiable GUD basis audit."""


class GUDOutputJacobianAuditShapeError(GUDOutputJacobianAuditError):
    """Geometry, role order, or a tensor shape is invalid."""


class GUDOutputJacobianAuditRankError(GUDOutputJacobianAuditError):
    """A requested candidate cannot retain its certified rank."""


class GUDOutputJacobianAuditFiniteError(GUDOutputJacobianAuditError):
    """A source or derived value contains NaN or infinity."""


class GUDOutputJacobianAuditLimitError(GUDOutputJacobianAuditError):
    """The explicit auxiliary-memory ceiling would be exceeded."""


class GUDOutputJacobianAuditDeterminismError(GUDOutputJacobianAuditError):
    """Repeated deterministic post-processing produced different bytes."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise GUDOutputJacobianAuditError(
            "Audit evidence must be canonical JSON."
        ) from error


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise GUDOutputJacobianAuditShapeError(f"{name} must be positive.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GUDOutputJacobianAuditShapeError(f"{name} must be positive.") from error
    if result <= 0:
        raise GUDOutputJacobianAuditShapeError(f"{name} must be positive.")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise GUDOutputJacobianAuditShapeError(f"{name} must be nonnegative.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GUDOutputJacobianAuditShapeError(
            f"{name} must be nonnegative."
        ) from error
    if result < 0:
        raise GUDOutputJacobianAuditShapeError(f"{name} must be nonnegative.")
    return result


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise GUDOutputJacobianAuditShapeError(f"{name} must be nonnegative.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise GUDOutputJacobianAuditShapeError(
            f"{name} must be nonnegative."
        ) from error
    if not math.isfinite(result) or result < 0.0:
        raise GUDOutputJacobianAuditShapeError(f"{name} must be nonnegative.")
    return result


def _derived_seed(seed: int, *parts: object) -> int:
    payload = ":".join(
        (GUD_OUTPUT_JACOBIAN_AUDIT_SCHEMA, str(seed), *(str(part) for part in parts))
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63)


def _validate_matrix_metadata(
    value: Any,
    name: str,
    *,
    allow_empty_axis: bool = False,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise GUDOutputJacobianAuditShapeError(f"{name} must be a torch tensor.")
    if value.layout != torch.strided or value.device.type == "meta":
        raise GUDOutputJacobianAuditShapeError(
            f"{name} must be a materialized dense-strided tensor."
        )
    if value.ndim != 2 or (not allow_empty_axis and 0 in value.shape):
        raise GUDOutputJacobianAuditShapeError(
            f"{name} must be a nonempty rank-two matrix."
        )
    if value.dtype not in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }:
        raise GUDOutputJacobianAuditShapeError(
            f"{name} must have a real floating-point dtype."
        )
    if value.is_conj() or value.is_neg():
        raise GUDOutputJacobianAuditShapeError(
            f"{name} cannot carry conjugate or negative view metadata."
        )
    return value


def _tensor_fingerprint(
    value: Tensor,
    name: str,
    *,
    chunk_bytes: int,
    allow_empty_axis: bool = False,
) -> dict[str, Any]:
    """Hash metadata and every element with bounded CPU staging.

    Each tensor receives its own digest.  No shared random projection and no
    ``lanes * rows * width`` coefficient table is used.
    """

    matrix = _validate_matrix_metadata(
        value, name, allow_empty_axis=allow_empty_axis
    ).detach()
    row_bytes = int(matrix.shape[1]) * matrix.element_size()
    if row_bytes > chunk_bytes:
        raise GUDOutputJacobianAuditLimitError(
            f"One {name} row exceeds the fingerprint staging bound."
        )
    rows_per_chunk = max(1, chunk_bytes // max(1, row_bytes))
    header = {
        "dtype": str(matrix.dtype),
        "shape": [int(axis) for axis in matrix.shape],
    }
    digest = hashlib.sha256(_canonical_json(header) + b"\0")
    for start in range(0, int(matrix.shape[0]), rows_per_chunk):
        staged = matrix[start : start + rows_per_chunk].to(device="cpu").contiguous()
        if not bool(torch.isfinite(staged).all()):
            raise GUDOutputJacobianAuditFiniteError(f"{name} is non-finite.")
        flat = staged.reshape(-1).view(torch.uint8)
        digest.update(memoryview(flat.numpy()))
    return {
        "name": name,
        **header,
        "numel": int(matrix.numel()),
        "nbytes": int(matrix.numel() * matrix.element_size()),
        "sha256": digest.hexdigest(),
        "every_element_bound": True,
        "separable_tensor_digest": True,
    }


@dataclass(frozen=True, slots=True)
class GUDRoleSpec:
    """One ordered native SwiGLU projection role."""

    role: GUDRole
    input_width: int
    output_width: int
    input_semantics: str
    output_semantics: str

    def __post_init__(self) -> None:
        if self.role not in GUD_ROLE_ORDER:
            raise GUDOutputJacobianAuditShapeError("role must be gate, up, or down.")
        object.__setattr__(
            self, "input_width", _positive_int(self.input_width, "input_width")
        )
        object.__setattr__(
            self, "output_width", _positive_int(self.output_width, "output_width")
        )
        for name in ("input_semantics", "output_semantics"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise GUDOutputJacobianAuditShapeError(f"{name} must be nonempty.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ordered_gud_role_specs(
    hidden_size: int,
    intermediate_size: int,
) -> tuple[GUDRoleSpec, GUDRoleSpec, GUDRoleSpec]:
    """Return the only accepted Gate -> Up -> Down role order."""

    hidden = _positive_int(hidden_size, "hidden_size")
    intermediate = _positive_int(intermediate_size, "intermediate_size")
    return (
        GUDRoleSpec(
            "gate",
            hidden,
            intermediate,
            "residual-hidden",
            "pre-silu-gate",
        ),
        GUDRoleSpec(
            "up",
            hidden,
            intermediate,
            "residual-hidden",
            "multiplicative-up",
        ),
        GUDRoleSpec(
            "down",
            intermediate,
            hidden,
            "post-swiglu-product",
            "residual-update",
        ),
    )


@dataclass(frozen=True, slots=True)
class GUDStageGeometry:
    """Generic dense-SwiGLU stage geometry."""

    logical_layers: int
    hidden_size: int
    intermediate_size: int

    def __post_init__(self) -> None:
        for name in ("logical_layers", "hidden_size", "intermediate_size"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))

    @property
    def role_specs(self) -> tuple[GUDRoleSpec, GUDRoleSpec, GUDRoleSpec]:
        return ordered_gud_role_specs(self.hidden_size, self.intermediate_size)

    def role_spec(self, role: GUDRole) -> GUDRoleSpec:
        for spec in self.role_specs:
            if spec.role == role:
                return spec
        raise GUDOutputJacobianAuditShapeError("Unknown GUD role.")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "ordered_role_specs": [spec.to_dict() for spec in self.role_specs],
        }


def validate_qwen_30_layer_stage_geometry(
    geometry: GUDStageGeometry,
) -> tuple[GUDRoleSpec, GUDRoleSpec, GUDRoleSpec]:
    """Fail closed unless this is the exact current 30-layer Qwen stage."""

    if not isinstance(geometry, GUDStageGeometry):
        raise GUDOutputJacobianAuditShapeError("geometry must be GUDStageGeometry.")
    expected = (
        QWEN_STAGE_LOGICAL_LAYERS,
        QWEN_STAGE_HIDDEN_SIZE,
        QWEN_STAGE_INTERMEDIATE_SIZE,
    )
    actual = (
        geometry.logical_layers,
        geometry.hidden_size,
        geometry.intermediate_size,
    )
    if actual != expected:
        raise GUDOutputJacobianAuditShapeError(
            "The exact stage requires 30 layers, hidden size 1536, and "
            "intermediate size 5120."
        )
    specs = geometry.role_specs
    if tuple(spec.role for spec in specs) != GUD_ROLE_ORDER:
        raise GUDOutputJacobianAuditShapeError(
            "The exact stage role order must be Gate, Up, Down."
        )
    return specs


@dataclass(frozen=True, slots=True)
class GUDJacobianBlock:
    """One role-local replay block; no model hook is retained."""

    output_delta: Tensor
    projection_input: Tensor
    semantic_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.semantic_identity, str) or not self.semantic_identity:
            raise GUDOutputJacobianAuditShapeError(
                "semantic_identity must be nonempty."
            )
        if len(self.semantic_identity.encode("utf-8")) > 64 * 1024:
            raise GUDOutputJacobianAuditShapeError(
                "semantic_identity exceeds 65536 UTF-8 bytes."
            )


@dataclass(frozen=True, slots=True)
class GUDSiteAuditInput:
    """All bounded evidence for one layer/role site."""

    layer_index: int
    role: GUDRole
    base_weight: Tensor
    inherited_left: Tensor
    inherited_right: Tensor
    blocks: tuple[GUDJacobianBlock, ...]
    forward_responses: tuple[Tensor, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "layer_index", _nonnegative_int(self.layer_index, "layer_index")
        )
        if self.role not in GUD_ROLE_ORDER:
            raise GUDOutputJacobianAuditShapeError("role must be gate, up, or down.")
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "forward_responses", tuple(self.forward_responses))
        if not self.blocks or any(
            not isinstance(block, GUDJacobianBlock) for block in self.blocks
        ):
            raise GUDOutputJacobianAuditShapeError(
                "blocks must be a nonempty tuple of GUDJacobianBlock values."
            )


@dataclass(frozen=True, slots=True)
class GUDOutputJacobianAuditConfig:
    """Deterministic ranks, gates, row bounds, and hard auxiliary limit."""

    rank: int = 8
    sketch_width: int = 16
    balanced_envelope_width: int = 32
    seed: int = 0
    max_blocks_per_site: int = 512
    max_rows_per_site: int = 4096
    min_normal_fraction: float = 0.0
    min_guided_over_random_ratio: float = 0.0
    fingerprint_chunk_bytes: int = DEFAULT_FINGERPRINT_CHUNK_BYTES
    max_bytes: int = HARD_MAX_AUXILIARY_BYTES

    def __post_init__(self) -> None:
        for name in (
            "rank",
            "sketch_width",
            "balanced_envelope_width",
            "max_blocks_per_site",
            "max_rows_per_site",
            "fingerprint_chunk_bytes",
            "max_bytes",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if isinstance(self.seed, bool):
            raise GUDOutputJacobianAuditShapeError("seed must be an integer.")
        try:
            seed = operator.index(self.seed)
        except (TypeError, ValueError, OverflowError) as error:
            raise GUDOutputJacobianAuditShapeError(
                "seed must be an integer."
            ) from error
        object.__setattr__(self, "seed", seed)
        for name in ("min_normal_fraction", "min_guided_over_random_ratio"):
            object.__setattr__(
                self, name, _finite_nonnegative(getattr(self, name), name)
            )
        if self.rank > self.sketch_width:
            raise GUDOutputJacobianAuditShapeError("rank cannot exceed sketch_width.")
        if self.balanced_envelope_width < 2 * self.rank:
            raise GUDOutputJacobianAuditShapeError(
                "balanced_envelope_width must reserve rank directions for both "
                "sensitivity and activation evidence."
            )
        if self.fingerprint_chunk_bytes > 4 * 1024 * 1024:
            raise GUDOutputJacobianAuditLimitError(
                "fingerprint_chunk_bytes cannot exceed 4 MiB."
            )
        if self.max_bytes > HARD_MAX_AUXILIARY_BYTES:
            raise GUDOutputJacobianAuditLimitError(
                "max_bytes exceeds the hard 256 MiB ceiling."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def sha256(self) -> str:
        return _canonical_sha256(
            {"schema": GUD_OUTPUT_JACOBIAN_AUDIT_SCHEMA, "config": self.to_dict()}
        )


@dataclass(frozen=True, slots=True)
class GUDSiteAuditReport:
    payload: Mapping[str, Any]

    @property
    def report_sha256(self) -> str:
        return str(self.payload["report_sha256"])

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True, slots=True)
class GUDSiteBasisResult:
    output_jacobian_right_rows: Tensor
    balanced_right_rows: Tensor
    procrustes_hybrid_right_rows: Tensor
    random_control_right_rows: Tensor
    report: GUDSiteAuditReport

    def candidates(self) -> Mapping[CandidateKind, Tensor]:
        return MappingProxyType(
            {
                "output-jacobian": self.output_jacobian_right_rows,
                "balanced": self.balanced_right_rows,
                "procrustes-hybrid": self.procrustes_hybrid_right_rows,
                "random-control": self.random_control_right_rows,
            }
        )


@dataclass(frozen=True, slots=True)
class GUDStageAuditReport:
    payload: Mapping[str, Any]

    @property
    def report_sha256(self) -> str:
        return str(self.payload["report_sha256"])

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


@dataclass(frozen=True, slots=True)
class GUDStageAuditResult:
    sites: tuple[GUDSiteBasisResult, ...]
    report: GUDStageAuditReport


def role_local_input_vjp(
    output_delta: Tensor,
    base_weight: Tensor,
    inherited_left: Tensor,
    inherited_right: Tensor,
) -> Tensor:
    """Return the exact local input VJP of ``W_base + L_inherited R_inherited``.

    For row-major deltas this is exactly::

        D @ W_base + (D @ L_inherited) @ R_inherited

    The function is intentionally role-local.  It does not use, register, or
    emulate a shared MLP-input hook.
    """

    delta = _validate_matrix_metadata(output_delta, "output_delta")
    weight = _validate_matrix_metadata(base_weight, "base_weight")
    left = _validate_matrix_metadata(
        inherited_left, "inherited_left", allow_empty_axis=True
    )
    right = _validate_matrix_metadata(
        inherited_right, "inherited_right", allow_empty_axis=True
    )
    rows, output_width = (int(delta.shape[0]), int(delta.shape[1]))
    input_width = int(weight.shape[1])
    latent = int(left.shape[1])
    if tuple(weight.shape) != (output_width, input_width):
        raise GUDOutputJacobianAuditShapeError(
            "base_weight must have shape [output_width, input_width]."
        )
    if tuple(left.shape) != (output_width, latent) or tuple(right.shape) != (
        latent,
        input_width,
    ):
        raise GUDOutputJacobianAuditShapeError(
            "Inherited factors do not match the role-local projection."
        )
    values = (delta, weight, left, right)
    if len({value.device for value in values}) != 1:
        raise GUDOutputJacobianAuditShapeError(
            "Role-local VJP tensors must be on one device."
        )
    if len({value.dtype for value in values}) != 1:
        raise GUDOutputJacobianAuditShapeError(
            "Role-local VJP tensors must have one floating dtype."
        )
    if rows <= 0:
        raise GUDOutputJacobianAuditShapeError("output_delta cannot be empty.")
    with torch.no_grad():
        result = role_local_input_vjp_prevalidated(
            delta.detach(), weight.detach(), left.detach(), right.detach()
        )
    if tuple(result.shape) != (rows, input_width):
        raise GUDOutputJacobianAuditShapeError(
            "Role-local input VJP has the wrong derived shape."
        )
    if not bool(torch.isfinite(result).all()):
        raise GUDOutputJacobianAuditFiniteError("Role-local input VJP is non-finite.")
    return result


def role_local_input_vjp_prevalidated(
    output_delta: Tensor,
    base_weight: Tensor,
    inherited_left: Tensor,
    inherited_right: Tensor,
) -> Tensor:
    """Hot-path local VJP after a collector has validated metadata.

    Unlike :func:`role_local_input_vjp`, this helper intentionally contains no
    finite scan or host synchronization.  Production replay collectors can
    call it from an output-gradient hook and authenticate/fail-close at the
    pass boundary.
    """

    result = output_delta @ base_weight
    if int(inherited_left.shape[1]):
        result = result + (output_delta @ inherited_left) @ inherited_right
    return result


def _site_shape_summary(
    site: GUDSiteAuditInput,
    spec: GUDRoleSpec,
    config: GUDOutputJacobianAuditConfig,
) -> dict[str, int]:
    if site.role != spec.role:
        raise GUDOutputJacobianAuditShapeError(
            "Site role disagrees with its ordered role specification."
        )
    weight = _validate_matrix_metadata(site.base_weight, "base_weight")
    left = _validate_matrix_metadata(
        site.inherited_left, "inherited_left", allow_empty_axis=True
    )
    right = _validate_matrix_metadata(
        site.inherited_right, "inherited_right", allow_empty_axis=True
    )
    latent = int(left.shape[1])
    if tuple(weight.shape) != (spec.output_width, spec.input_width):
        raise GUDOutputJacobianAuditShapeError(
            "base_weight disagrees with the role geometry."
        )
    if tuple(left.shape) != (spec.output_width, latent) or tuple(right.shape) != (
        latent,
        spec.input_width,
    ):
        raise GUDOutputJacobianAuditShapeError(
            "Inherited factors disagree with the role geometry."
        )
    if latent + config.rank > spec.input_width:
        raise GUDOutputJacobianAuditRankError(
            "Inherited and requested right spaces exceed the role input width."
        )
    if config.rank > spec.output_width:
        raise GUDOutputJacobianAuditRankError(
            "Requested rank exceeds the role output width."
        )
    if len(site.blocks) > config.max_blocks_per_site:
        raise GUDOutputJacobianAuditLimitError(
            "Site block count exceeds its preregistered bound."
        )
    total_rows = 0
    maximum_rows = 0
    identities: set[str] = set()
    for index, block in enumerate(site.blocks):
        delta = _validate_matrix_metadata(block.output_delta, f"blocks[{index}].delta")
        inputs = _validate_matrix_metadata(
            block.projection_input, f"blocks[{index}].inputs"
        )
        rows = int(delta.shape[0])
        if tuple(delta.shape) != (rows, spec.output_width) or tuple(inputs.shape) != (
            rows,
            spec.input_width,
        ):
            raise GUDOutputJacobianAuditShapeError(
                "A block disagrees with its role-local output/input widths."
            )
        if block.semantic_identity in identities:
            raise GUDOutputJacobianAuditShapeError(
                "Block semantic identities must be unique within a site."
            )
        identities.add(block.semantic_identity)
        if (
            delta.device != weight.device
            or delta.device != left.device
            or delta.device != right.device
            or delta.dtype != weight.dtype
            or delta.dtype != left.dtype
            or delta.dtype != right.dtype
        ):
            raise GUDOutputJacobianAuditShapeError(
                "D, W_base, L_inherited, and R_inherited must share device/dtype."
            )
        total_rows += rows
        maximum_rows = max(maximum_rows, rows)
    if total_rows > config.max_rows_per_site:
        raise GUDOutputJacobianAuditLimitError(
            "Site replay rows exceed their preregistered bound."
        )
    forward_rows = 0
    maximum_forward_rows = 0
    for index, value in enumerate(site.forward_responses):
        matrix = _validate_matrix_metadata(value, f"forward_responses[{index}]")
        if int(matrix.shape[1]) != spec.input_width:
            raise GUDOutputJacobianAuditShapeError(
                "A forward response has the wrong role-local input width."
            )
        forward_rows += int(matrix.shape[0])
        maximum_forward_rows = max(maximum_forward_rows, int(matrix.shape[0]))
    if total_rows + forward_rows > config.max_rows_per_site:
        raise GUDOutputJacobianAuditLimitError(
            "Balanced rows exceed their preregistered per-site bound."
        )
    return {
        "block_count": len(site.blocks),
        "activation_rows": total_rows,
        "sensitivity_rows": total_rows,
        "forward_rows": forward_rows,
        "maximum_block_rows": maximum_rows,
        "maximum_forward_rows": maximum_forward_rows,
        "inherited_rank": latent,
    }


def _sketch_upper_bound(
    spec: GUDRoleSpec,
    summary: Mapping[str, int],
    config: GUDOutputJacobianAuditConfig,
) -> int:
    out_features = spec.output_width
    in_features = spec.input_width
    inherited = int(summary["inherited_rank"])
    width = config.sketch_width
    rank = config.rank
    rows = int(summary["maximum_block_rows"])
    range_width = min(width, in_features - inherited)
    occupied_bytes = 4 * inherited * (out_features + in_features)
    pass1_persistent = (
        occupied_bytes + 4 * out_features * width + 8 * in_features * width
    )
    pass2_persistent = occupied_bytes + 4 * (in_features + out_features) * (
        range_width + rank
    )
    factor_analysis = (
        8 * inherited * (out_features + in_features)
        + 32 * inherited * inherited
        + 16
        * (
            out_features * min(out_features, inherited)
            + in_features * min(in_features, inherited)
            + 2 * inherited * inherited
        )
    )
    pass1_block = 4 * rows * (
        3 * out_features + 3 * in_features + 2 * inherited
    ) + 4 * (rows * width + in_features * width)
    transition = 24 * in_features * width + 16 * in_features * rank + 8 * width * width
    pass2_block = 4 * rows * (
        3 * out_features + 3 * in_features + 2 * inherited
    ) + 4 * (rows * (range_width + rank) + out_features * max(range_width, rank))
    finalize = (
        16 * out_features * range_width
        + 8 * range_width * range_width
        + 8 * range_width
        + 8 * in_features * (range_width + rank)
        + 12 * rank * in_features
    )
    return max(pass1_persistent, pass2_persistent) + max(
        factor_analysis,
        pass1_block,
        transition,
        pass2_block,
        finalize,
    )


def _site_working_memory(
    spec: GUDRoleSpec,
    summary: Mapping[str, int],
    config: GUDOutputJacobianAuditConfig,
) -> dict[str, int]:
    width = spec.input_width
    rank = config.rank
    activation_rows = int(summary["activation_rows"])
    sensitivity_rows = int(summary["sensitivity_rows"])
    forward_rows = int(summary["forward_rows"])
    maximum_source_rows = max(
        int(summary["maximum_block_rows"]),
        int(summary["maximum_forward_rows"]),
    )
    envelope = min(
        config.balanced_envelope_width, width - int(summary["inherited_rank"])
    )
    reduced_rows = activation_rows + sensitivity_rows + forward_rows
    output_sketch = _sketch_upper_bound(spec, summary, config)
    vjp_storage = 8 * sensitivity_rows * width
    role_local_vjp_peak = (
        8
        * int(summary["maximum_block_rows"])
        * (2 * width + int(summary["inherited_rank"]))
    )
    full_width_source_staging = 8 * maximum_source_rows * width
    envelope_basis = 8 * envelope * width
    reduced_snapshot_storage = 8 * reduced_rows * envelope
    # ``balanced_krylov`` is invoked only in the reduced envelope.  This
    # bound covers its input copies, empirical blocks, compact SVD factors,
    # reduced projector/products, and LAPACK-sized result buffers.
    balanced_reduced_buffers = 8 * (
        20 * reduced_rows * envelope + 96 * envelope * envelope
    )
    candidate_postprocessing = 8 * (12 * rank * width + 12 * rank * rank)
    fingerprint_staging = 2 * config.fingerprint_chunk_bytes
    peak = (
        output_sketch
        + vjp_storage
        + role_local_vjp_peak
        + full_width_source_staging
        + envelope_basis
        + reduced_snapshot_storage
        + balanced_reduced_buffers
        + candidate_postprocessing
        + fingerprint_staging
    )
    return {
        "output_jacobian_sketch_upper_bound_bytes": output_sketch,
        "role_local_input_vjp_storage_bytes": vjp_storage,
        "role_local_input_vjp_peak_bytes": role_local_vjp_peak,
        "largest_full_width_source_staging_bytes": full_width_source_staging,
        "balanced_envelope_basis_bytes": envelope_basis,
        "balanced_reduced_snapshot_storage_bytes": reduced_snapshot_storage,
        "balanced_reduced_internal_buffer_upper_bound_bytes": (
            balanced_reduced_buffers
        ),
        "candidate_postprocessing_upper_bound_bytes": candidate_postprocessing,
        "separable_fingerprint_staging_bytes": fingerprint_staging,
        "site_working_peak_bytes": peak,
        "balanced_envelope_width_upper_bound": envelope,
        "full_width_balanced_projector_bytes": 0,
        "full_lanes_rows_width_fingerprint_table_bytes": 0,
    }


def plan_gud_auxiliary_memory(
    geometry: GUDStageGeometry,
    sites: Sequence[GUDSiteAuditInput],
    config: GUDOutputJacobianAuditConfig,
) -> dict[str, Any]:
    """Return and authenticate the conservative sequential-site memory plan."""

    if not isinstance(geometry, GUDStageGeometry) or not isinstance(
        config, GUDOutputJacobianAuditConfig
    ):
        raise GUDOutputJacobianAuditShapeError(
            "geometry and config have invalid types."
        )
    site_values = tuple(sites)
    if not site_values:
        raise GUDOutputJacobianAuditShapeError("sites cannot be empty.")
    summaries: list[dict[str, int]] = []
    working: list[dict[str, int]] = []
    output_pack_bytes = 0
    for site in site_values:
        if not isinstance(site, GUDSiteAuditInput):
            raise GUDOutputJacobianAuditShapeError(
                "Every site must be GUDSiteAuditInput."
            )
        role_spec = geometry.role_spec(site.role)
        summary = _site_shape_summary(site, role_spec, config)
        summaries.append(summary)
        working.append(_site_working_memory(role_spec, summary, config))
        output_pack_bytes += (
            len(_CANDIDATE_ORDER) * config.rank * role_spec.input_width * 4
        )
    largest_site_working = max(row["site_working_peak_bytes"] for row in working)
    peak = output_pack_bytes + largest_site_working
    payload: dict[str, Any] = {
        "schema": f"{GUD_OUTPUT_JACOBIAN_AUDIT_SCHEMA}:memory-plan-v1",
        "site_count": len(site_values),
        "four_candidate_output_pack_bytes": output_pack_bytes,
        "largest_sequential_site_working_peak_bytes": largest_site_working,
        "accounted_auxiliary_peak_bytes": peak,
        "max_auxiliary_bytes": config.max_bytes,
        "hard_max_auxiliary_bytes": HARD_MAX_AUXILIARY_BYTES,
        "within_limit": peak <= config.max_bytes <= HARD_MAX_AUXILIARY_BYTES,
        "processing_mode": "one-role-site-at-a-time",
        "balanced_mode": "full-width-envelope-then-balanced-center-false",
        "balanced_centered": False,
        "full_width_balanced_projector_bytes": 0,
        "full_lanes_rows_width_fingerprint_table_bytes": 0,
        "fingerprint_mode": "separable-streamed-exact-sha256-per-tensor",
        "fingerprint_binds_every_element": True,
        "site_shape_summaries": summaries,
        "site_working_plans": working,
        "memory_scope": {
            "includes": [
                "output-jacobian-sketch-buffers",
                "role-local-input-vjp-results",
                "bounded-full-width-row-staging",
                "balanced-envelope-and-reduced-balanced-buffers",
                "all-four-final-candidate-packs",
                "separable-full-element-hash-staging",
            ],
            "excludes": [
                "caller-owned-model-weights",
                "caller-owned-replay-blocks",
                "framework-allocator-reservations",
                "python-object-overhead",
            ],
        },
    }
    payload["memory_plan_sha256"] = _canonical_sha256(payload)
    if not payload["within_limit"]:
        raise GUDOutputJacobianAuditLimitError(
            "GUD audit auxiliary tensors exceed the configured/hard byte ceiling."
        )
    return payload


def _row_orthonormal_basis(
    rows: Tensor,
    *,
    maximum_rows: int | None = None,
    require_rows: int = 0,
) -> Tensor:
    matrix = rows.detach().to(device="cpu", dtype=torch.float64)
    if matrix.ndim != 2:
        raise GUDOutputJacobianAuditShapeError("Basis source must be rank two.")
    limit = int(matrix.shape[0]) if maximum_rows is None else maximum_rows
    selected: list[Tensor] = []
    scale = float(torch.linalg.vector_norm(matrix).item())
    tolerance = max(
        torch.finfo(torch.float64).eps * max(matrix.shape) * max(scale, 1.0) * 256.0,
        1e-13,
    )
    for row in matrix:
        candidate = row.clone()
        for _ in range(2):
            for existing in selected:
                candidate -= torch.dot(candidate, existing) * existing
        norm = float(torch.linalg.vector_norm(candidate).item())
        if norm <= tolerance:
            continue
        candidate.div_(norm)
        pivot = int(torch.argmax(candidate.abs()).item())
        if float(candidate[pivot].item()) < 0.0:
            candidate.neg_()
        selected.append(candidate)
        if len(selected) == limit:
            break
    if len(selected) < require_rows:
        raise GUDOutputJacobianAuditRankError(
            f"Row space has rank {len(selected)}, below required rank {require_rows}."
        )
    if not selected:
        return torch.empty((0, int(matrix.shape[1])), dtype=torch.float64)
    result = torch.stack(selected)
    if not bool(torch.isfinite(result).all()):
        raise GUDOutputJacobianAuditFiniteError("Row basis became non-finite.")
    return result


def _project_candidate(
    candidate: Tensor,
    occupied: Tensor,
    rank: int,
) -> Tensor:
    rows = candidate.detach().to(device="cpu", dtype=torch.float64)
    if rows.ndim != 2 or int(rows.shape[0]) != rank:
        raise GUDOutputJacobianAuditShapeError(
            "A candidate must have exactly rank rows."
        )
    if int(occupied.shape[1]) != int(rows.shape[1]):
        raise GUDOutputJacobianAuditShapeError(
            "Occupied and candidate input widths differ."
        )
    if occupied.numel():
        for _ in range(2):
            rows = rows - (rows @ occupied.T) @ occupied
    result64 = _row_orthonormal_basis(rows, maximum_rows=rank, require_rows=rank)
    if occupied.numel():
        for _ in range(2):
            result64 = result64 - (result64 @ occupied.T) @ occupied
        result64 = _row_orthonormal_basis(
            result64, maximum_rows=rank, require_rows=rank
        )
    gram_error = float(
        (result64 @ result64.T - torch.eye(rank, dtype=torch.float64)).abs().max()
    )
    exclusion_error = (
        float((result64 @ occupied.T).abs().max()) if occupied.numel() else 0.0
    )
    if gram_error > 1e-10 or exclusion_error > 1e-10:
        raise GUDOutputJacobianAuditRankError(
            "Candidate could not be row-orthonormalized outside inherited space."
        )
    result = result64.to(dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise GUDOutputJacobianAuditFiniteError("Candidate became non-finite.")
    return result


def _cpu_float64(value: Tensor) -> Tensor:
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise GUDOutputJacobianAuditFiniteError(
            "Balanced source staging became non-finite."
        )
    return result


def _extend_envelope(
    selected: list[Tensor],
    matrices: Sequence[Tensor],
    occupied: Tensor,
    *,
    target_additions: int,
    total_limit: int,
) -> int:
    start_count = len(selected)
    for source in matrices:
        staged = _cpu_float64(source)
        for row in staged:
            candidate = row.clone()
            for _ in range(2):
                if occupied.numel():
                    candidate -= (candidate @ occupied.T) @ occupied
                for existing in selected:
                    candidate -= torch.dot(candidate, existing) * existing
            norm = float(torch.linalg.vector_norm(candidate).item())
            tolerance = max(
                1e-13,
                torch.finfo(torch.float64).eps
                * max(candidate.numel(), 1)
                * max(float(torch.linalg.vector_norm(row).item()), 1.0)
                * 256.0,
            )
            if norm <= tolerance:
                continue
            candidate.div_(norm)
            pivot = int(torch.argmax(candidate.abs()).item())
            if float(candidate[pivot].item()) < 0.0:
                candidate.neg_()
            selected.append(candidate)
            if (
                len(selected) - start_count >= target_additions
                or len(selected) >= total_limit
            ):
                return len(selected) - start_count
    return len(selected) - start_count


def _balanced_envelope(
    activation_blocks: Sequence[Tensor],
    sensitivity_blocks: Sequence[Tensor],
    forward_blocks: Sequence[Tensor],
    occupied: Tensor,
    config: GUDOutputJacobianAuditConfig,
) -> Tensor:
    width = int(activation_blocks[0].shape[1])
    limit = min(config.balanced_envelope_width, width - int(occupied.shape[0]))
    selected: list[Tensor] = []
    sensitivity_added = _extend_envelope(
        selected,
        sensitivity_blocks,
        occupied,
        target_additions=config.rank,
        total_limit=limit,
    )
    if sensitivity_added < config.rank:
        raise GUDOutputJacobianAuditRankError(
            "Role-local input VJPs cannot support the requested balanced rank."
        )
    activation_added = _extend_envelope(
        selected,
        activation_blocks,
        occupied,
        target_additions=config.rank,
        total_limit=limit,
    )
    if activation_added < config.rank:
        raise GUDOutputJacobianAuditRankError(
            "Projection inputs cannot support the requested balanced rank."
        )
    remaining = limit - len(selected)
    if remaining:
        _extend_envelope(
            selected,
            (*sensitivity_blocks, *activation_blocks, *forward_blocks),
            occupied,
            target_additions=remaining,
            total_limit=limit,
        )
    envelope = torch.stack(selected)
    error = float(
        (envelope @ envelope.T - torch.eye(len(selected), dtype=torch.float64))
        .abs()
        .max()
    )
    if error > 1e-10:
        raise GUDOutputJacobianAuditRankError(
            "Balanced envelope is not row-orthonormal."
        )
    return envelope


def finalize_streaming_gud_envelope(
    activation_row_sketch: Tensor,
    sensitivity_row_sketch: Tensor,
    inherited_right: Tensor,
    config: GUDOutputJacobianAuditConfig,
) -> Tensor:
    """Finalize bounded pass-one row sketches into a CPU FP64 envelope."""

    activations = _validate_matrix_metadata(
        activation_row_sketch, "activation_row_sketch"
    )
    sensitivities = _validate_matrix_metadata(
        sensitivity_row_sketch, "sensitivity_row_sketch"
    )
    inherited = _validate_matrix_metadata(
        inherited_right, "inherited_right", allow_empty_axis=True
    )
    if (
        int(activations.shape[1]) != int(sensitivities.shape[1])
        or int(inherited.shape[1]) != int(activations.shape[1])
    ):
        raise GUDOutputJacobianAuditShapeError(
            "Streaming row sketches and inherited right factors must share width."
        )
    occupied = _row_orthonormal_basis(inherited, require_rows=0)
    return _balanced_envelope(
        (activations,),
        (sensitivities,),
        (),
        occupied,
        config,
    )


def _reduced_groups(
    values: Sequence[Tensor], envelope: Tensor
) -> tuple[np.ndarray, ...]:
    result: list[np.ndarray] = []
    for value in values:
        reduced = _cpu_float64(value) @ envelope.T
        if not bool(torch.isfinite(reduced).all()):
            raise GUDOutputJacobianAuditFiniteError(
                "A reduced balanced snapshot is non-finite."
            )
        result.append(reduced.numpy().copy())
    return tuple(result)


def _procrustes_hybrid(jacobian: Tensor, balanced: Tensor) -> Tensor:
    left = jacobian.detach().to(device="cpu", dtype=torch.float64)
    right = balanced.detach().to(device="cpu", dtype=torch.float64)

    def build() -> Tensor:
        cross = left @ right.T
        u, singular, vh = torch.linalg.svd(cross, full_matrices=False)
        if not bool(torch.isfinite(singular).all()):
            raise GUDOutputJacobianAuditFiniteError(
                "Procrustes alignment became non-finite."
            )
        return (left + (u @ vh) @ right).contiguous()

    first = build()
    second = build()
    if not torch.equal(first, second):
        raise GUDOutputJacobianAuditDeterminismError(
            "Repeated Procrustes alignment produced different bytes."
        )
    return first


def _candidate_manifest(
    candidates: Mapping[CandidateKind, Tensor],
    config: GUDOutputJacobianAuditConfig,
) -> list[dict[str, Any]]:
    rows = []
    for kind in _CANDIDATE_ORDER:
        fingerprint = _tensor_fingerprint(
            candidates[kind],
            f"candidate:{kind}",
            chunk_bytes=config.fingerprint_chunk_bytes,
        )
        rows.append({"candidate": kind, **fingerprint})
    return rows


def _input_manifest(
    site: GUDSiteAuditInput,
    vjps: Sequence[Tensor],
    config: GUDOutputJacobianAuditConfig,
) -> dict[str, Any]:
    chunk = config.fingerprint_chunk_bytes
    blocks = []
    for index, (block, vjp) in enumerate(zip(site.blocks, vjps, strict=True)):
        blocks.append(
            {
                "index": index,
                "semantic_identity": block.semantic_identity,
                "output_delta": _tensor_fingerprint(
                    block.output_delta,
                    f"blocks[{index}].output_delta",
                    chunk_bytes=chunk,
                ),
                "projection_input": _tensor_fingerprint(
                    block.projection_input,
                    f"blocks[{index}].projection_input",
                    chunk_bytes=chunk,
                ),
                "role_local_input_vjp": _tensor_fingerprint(
                    vjp, f"blocks[{index}].role_local_input_vjp", chunk_bytes=chunk
                ),
            }
        )
    forward = [
        _tensor_fingerprint(value, f"forward_responses[{index}]", chunk_bytes=chunk)
        for index, value in enumerate(site.forward_responses)
    ]
    payload = {
        "base_weight": _tensor_fingerprint(
            site.base_weight, "base_weight", chunk_bytes=chunk
        ),
        "inherited_left": _tensor_fingerprint(
            site.inherited_left,
            "inherited_left",
            chunk_bytes=chunk,
            allow_empty_axis=True,
        ),
        "inherited_right": _tensor_fingerprint(
            site.inherited_right,
            "inherited_right",
            chunk_bytes=chunk,
            allow_empty_axis=True,
        ),
        "blocks": blocks,
        "forward_responses": forward,
        "fingerprint_mode": "separable-streamed-exact-sha256-per-tensor",
        "every_element_bound": True,
        "shared_projection_fingerprint_used": False,
        "full_lanes_rows_width_table_used": False,
    }
    payload["input_manifest_sha256"] = _canonical_sha256(payload)
    return payload


def _build_site(
    site: GUDSiteAuditInput,
    spec: GUDRoleSpec,
    config: GUDOutputJacobianAuditConfig,
    memory_plan: Mapping[str, Any],
) -> GUDSiteBasisResult:
    summary = _site_shape_summary(site, spec, config)
    vjps = tuple(
        role_local_input_vjp(
            block.output_delta,
            site.base_weight,
            site.inherited_left,
            site.inherited_right,
        )
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        for block in site.blocks
    )
    input_manifest = _input_manifest(site, vjps, config)
    sketch_spec = OutputJacobianSketchSpec(
        out_features=spec.output_width,
        in_features=spec.input_width,
        rank=config.rank,
        sketch_width=config.sketch_width,
        seed=_derived_seed(config.seed, site.layer_index, site.role, "jacobian"),
        accumulator_device="cpu",
        max_bytes=config.max_bytes,
        min_normal_fraction=config.min_normal_fraction,
        min_guided_over_random_ratio=config.min_guided_over_random_ratio,
    )
    try:
        sketch = TwoPassOutputJacobianSketch(
            sketch_spec,
            site.inherited_left,
            site.inherited_right,
            accumulator_device="cpu",
        )
        identities: list[OutputJacobianBlockIdentity] = []
        for block, evidence in zip(site.blocks, input_manifest["blocks"], strict=True):
            delta_sha256 = evidence["output_delta"]["sha256"]
            inputs_sha256 = evidence["projection_input"]["sha256"]
            identity = OutputJacobianBlockIdentity(
                semantic_identity=block.semantic_identity,
                numeric_fingerprint_sha256=_canonical_sha256(
                    {
                        "delta_sha256": delta_sha256,
                        "inputs_sha256": inputs_sha256,
                    }
                ),
                delta_sha256=delta_sha256,
                inputs_sha256=inputs_sha256,
            )
            identities.append(identity)
            sketch.add_pass1(
                block.output_delta,
                block.projection_input,
                block_identity=identity,
            )
        sketch.finish_pass1()
        for block, identity in zip(site.blocks, identities, strict=True):
            sketch.add_pass2(
                block.output_delta,
                block.projection_input,
                block_identity=identity,
            )
        sketch_result = sketch.finalize()
    except OutputJacobianSketchError as error:
        raise GUDOutputJacobianAuditError(
            "The reused output-Jacobian sketch rejected this site."
        ) from error
    occupied = _row_orthonormal_basis(site.inherited_right, require_rows=0)
    activation_blocks = tuple(block.projection_input for block in site.blocks)
    envelope = _balanced_envelope(
        activation_blocks,
        vjps,
        site.forward_responses,
        occupied,
        config,
    )
    reduced_activations = _reduced_groups(activation_blocks, envelope)
    reduced_sensitivities = _reduced_groups(vjps, envelope)
    reduced_forward = _reduced_groups(site.forward_responses, envelope)
    try:
        balanced_basis = build_balanced_krylov_basis(
            reduced_activations,
            rank=config.rank,
            forward_responses=(reduced_forward or None),
            output_sensitivities=reduced_sensitivities,
            center=False,
            allow_svd_fallback=False,
        )
    except BalancedKrylovError as error:
        raise GUDOutputJacobianAuditRankError(
            "The reused center=False balanced basis rejected this site."
        ) from error
    if balanced_basis.report.centered_activations is not False:
        raise GUDOutputJacobianAuditError(
            "The balanced dependency did not honor center=False."
        )
    lifted_balanced = envelope.T @ torch.from_numpy(
        np.array(balanced_basis.decoder, dtype=np.float64, order="C", copy=True)
    )
    jacobian_rows = _project_candidate(
        sketch_result.guided_right_rows, occupied, config.rank
    )
    balanced_rows = _project_candidate(lifted_balanced.T, occupied, config.rank)
    random_rows = _project_candidate(
        sketch_result.random_right_rows, occupied, config.rank
    )
    hybrid_source = _procrustes_hybrid(jacobian_rows, balanced_rows)
    hybrid_rows = _project_candidate(hybrid_source, occupied, config.rank)
    # Repeat the cheap deterministic projection/alignment path and compare
    # exact FP32 bytes.  Dependency reports separately bind their own seeds and
    # provenance.
    repeated = {
        "output-jacobian": _project_candidate(
            sketch_result.guided_right_rows, occupied, config.rank
        ),
        "balanced": _project_candidate(lifted_balanced.T, occupied, config.rank),
        "procrustes-hybrid": _project_candidate(
            _procrustes_hybrid(jacobian_rows, balanced_rows), occupied, config.rank
        ),
        "random-control": _project_candidate(
            sketch_result.random_right_rows, occupied, config.rank
        ),
    }
    candidates: dict[CandidateKind, Tensor] = {
        "output-jacobian": jacobian_rows,
        "balanced": balanced_rows,
        "procrustes-hybrid": hybrid_rows,
        "random-control": random_rows,
    }
    if any(
        not torch.equal(candidates[kind], repeated[kind]) for kind in _CANDIDATE_ORDER
    ):
        raise GUDOutputJacobianAuditDeterminismError(
            "Repeated candidate construction produced different bytes."
        )
    candidate_manifest = _candidate_manifest(candidates, config)
    candidate_manifest_sha256 = _canonical_sha256(candidate_manifest)
    occupied_manifest = _tensor_fingerprint(
        occupied,
        "occupied_inherited_right_row_basis",
        chunk_bytes=config.fingerprint_chunk_bytes,
        allow_empty_axis=True,
    )
    envelope_manifest = _tensor_fingerprint(
        envelope,
        "balanced_envelope",
        chunk_bytes=config.fingerprint_chunk_bytes,
    )
    report_payload: dict[str, Any] = {
        "schema": GUD_OUTPUT_JACOBIAN_AUDIT_SCHEMA,
        "candidate_schema": GUD_CANDIDATE_BASIS_SCHEMA,
        "layer_index": site.layer_index,
        "role": site.role,
        "role_spec": spec.to_dict(),
        "config_sha256": config.sha256(),
        "shape_summary": summary,
        "input_manifest": input_manifest,
        "input_manifest_sha256": input_manifest["input_manifest_sha256"],
        "occupied_inherited_right_row_rank": int(occupied.shape[0]),
        "occupied_inherited_right_row_basis": occupied_manifest,
        "balanced_envelope_width": int(envelope.shape[0]),
        "balanced_envelope": envelope_manifest,
        "balanced_centered": False,
        "balanced_dependency_report": balanced_basis.report.as_dict(),
        "output_jacobian_dependency_report": sketch_result.report.to_dict(),
        "candidate_manifest": candidate_manifest,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "candidate_order": list(_CANDIDATE_ORDER),
        "equal_rank_random_control": True,
        "all_candidates_projected_out_of_inherited_right_row_space": True,
        "all_candidates_row_orthonormal": True,
        "role_local_input_vjp_formula": ("D@W_base + (D@L_inherited)@R_inherited"),
        "shared_input_hook_used": False,
        "internal_weight_matching_used": False,
        "activation_target_matching_used": False,
        "optimizer_created": False,
        "backward_invoked_by_core": False,
        "determinism_verified_by_exact_repeat": True,
        "separable_full_element_fingerprints": True,
        "full_lanes_rows_width_fingerprint_table_used": False,
        "memory_plan_sha256": memory_plan["memory_plan_sha256"],
        "accounted_auxiliary_peak_bytes": memory_plan["accounted_auxiliary_peak_bytes"],
        "max_auxiliary_bytes": memory_plan["max_auxiliary_bytes"],
    }
    report_payload["report_sha256"] = _canonical_sha256(report_payload)
    report = GUDSiteAuditReport(MappingProxyType(dict(report_payload)))
    return GUDSiteBasisResult(
        output_jacobian_right_rows=jacobian_rows,
        balanced_right_rows=balanced_rows,
        procrustes_hybrid_right_rows=hybrid_rows,
        random_control_right_rows=random_rows,
        report=report,
    )


def finalize_streamed_gud_site_basis(
    *,
    layer_index: int,
    role_spec: GUDRoleSpec,
    inherited_right: Tensor,
    output_jacobian_result: OutputJacobianSketchResult,
    balanced_envelope: Tensor,
    reduced_activation_rows: Tensor,
    reduced_sensitivity_rows: Tensor,
    config: GUDOutputJacobianAuditConfig,
    source_manifest: Mapping[str, Any],
    memory_plan_sha256: str,
) -> GUDSiteBasisResult:
    """Finalize one site from real streaming replay evidence.

    The replay owner supplies a completed two-pass sketch, a pass-one envelope,
    and only bounded pass-two coordinates.  No full-width replay table is
    reconstructed here.
    """

    layer = _nonnegative_int(layer_index, "layer_index")
    if not isinstance(role_spec, GUDRoleSpec):
        raise GUDOutputJacobianAuditShapeError("role_spec has the wrong type.")
    if not isinstance(output_jacobian_result, OutputJacobianSketchResult):
        raise GUDOutputJacobianAuditShapeError(
            "output_jacobian_result has the wrong type."
        )
    if not isinstance(config, GUDOutputJacobianAuditConfig):
        raise GUDOutputJacobianAuditShapeError("config has the wrong type.")
    if (
        not isinstance(memory_plan_sha256, str)
        or len(memory_plan_sha256) != 64
        or any(character not in "0123456789abcdef" for character in memory_plan_sha256)
    ):
        raise GUDOutputJacobianAuditShapeError(
            "memory_plan_sha256 must be a lowercase SHA-256 digest."
        )
    if not isinstance(source_manifest, Mapping):
        raise GUDOutputJacobianAuditShapeError("source_manifest must be a mapping.")
    envelope = _validate_matrix_metadata(
        balanced_envelope, "balanced_envelope"
    ).detach().to(device="cpu", dtype=torch.float64).contiguous()
    activations = _validate_matrix_metadata(
        reduced_activation_rows, "reduced_activation_rows"
    ).detach().to(device="cpu", dtype=torch.float64).contiguous()
    sensitivities = _validate_matrix_metadata(
        reduced_sensitivity_rows, "reduced_sensitivity_rows"
    ).detach().to(device="cpu", dtype=torch.float64).contiguous()
    envelope_width = int(envelope.shape[0])
    if (
        int(envelope.shape[1]) != role_spec.input_width
        or int(activations.shape[1]) != envelope_width
        or int(sensitivities.shape[1]) != envelope_width
        or envelope_width < config.rank
    ):
        raise GUDOutputJacobianAuditShapeError(
            "Streamed balanced evidence disagrees with the role/envelope geometry."
        )
    for name, value in (
        ("balanced_envelope", envelope),
        ("reduced_activation_rows", activations),
        ("reduced_sensitivity_rows", sensitivities),
    ):
        if not bool(torch.isfinite(value).all()):
            raise GUDOutputJacobianAuditFiniteError(f"{name} is non-finite.")
    envelope_gram_error = float(
        (
            envelope @ envelope.T
            - torch.eye(envelope_width, dtype=torch.float64)
        )
        .abs()
        .max()
    )
    if envelope_gram_error > 1e-10:
        raise GUDOutputJacobianAuditRankError(
            "The streamed balanced envelope is not row-orthonormal."
        )
    occupied = _row_orthonormal_basis(inherited_right, require_rows=0)
    occupied_error = (
        float((envelope @ occupied.T).abs().max()) if occupied.numel() else 0.0
    )
    if occupied_error > 1e-10:
        raise GUDOutputJacobianAuditRankError(
            "The streamed envelope overlaps inherited right-row space."
        )
    try:
        balanced_basis = build_balanced_krylov_basis(
            activations.numpy().copy(),
            rank=config.rank,
            output_sensitivities=sensitivities.numpy().copy(),
            center=False,
            allow_svd_fallback=False,
        )
    except BalancedKrylovError as error:
        raise GUDOutputJacobianAuditRankError(
            "The streamed center=False balanced evidence is uncertifiable."
        ) from error
    lifted_balanced = envelope.T @ torch.from_numpy(
        np.array(balanced_basis.decoder, dtype=np.float64, order="C", copy=True)
    )
    jacobian_rows = _project_candidate(
        output_jacobian_result.guided_right_rows, occupied, config.rank
    )
    balanced_rows = _project_candidate(lifted_balanced.T, occupied, config.rank)
    random_rows = _project_candidate(
        output_jacobian_result.random_right_rows, occupied, config.rank
    )
    hybrid_rows = _project_candidate(
        _procrustes_hybrid(jacobian_rows, balanced_rows), occupied, config.rank
    )
    candidates: dict[CandidateKind, Tensor] = {
        "output-jacobian": jacobian_rows,
        "balanced": balanced_rows,
        "procrustes-hybrid": hybrid_rows,
        "random-control": random_rows,
    }
    repeated = {
        "output-jacobian": _project_candidate(
            output_jacobian_result.guided_right_rows, occupied, config.rank
        ),
        "balanced": _project_candidate(lifted_balanced.T, occupied, config.rank),
        "procrustes-hybrid": _project_candidate(
            _procrustes_hybrid(jacobian_rows, balanced_rows), occupied, config.rank
        ),
        "random-control": _project_candidate(
            output_jacobian_result.random_right_rows, occupied, config.rank
        ),
    }
    if any(
        not torch.equal(candidates[kind], repeated[kind])
        for kind in _CANDIDATE_ORDER
    ):
        raise GUDOutputJacobianAuditDeterminismError(
            "Repeated streamed candidate construction produced different bytes."
        )
    candidate_manifest = _candidate_manifest(candidates, config)
    source_payload = dict(source_manifest)
    source_manifest_sha256 = _canonical_sha256(source_payload)
    report_payload: dict[str, Any] = {
        "schema": GUD_OUTPUT_JACOBIAN_AUDIT_SCHEMA,
        "candidate_schema": GUD_CANDIDATE_BASIS_SCHEMA,
        "collection_mode": "streamed-real-model-loss-replay",
        "layer_index": layer,
        "role": role_spec.role,
        "role_spec": role_spec.to_dict(),
        "config_sha256": config.sha256(),
        "source_manifest": source_payload,
        "source_manifest_sha256": source_manifest_sha256,
        "output_jacobian_dependency_report": (
            output_jacobian_result.report.to_dict()
        ),
        "balanced_dependency_report": balanced_basis.report.as_dict(),
        "balanced_centered": False,
        "balanced_envelope_width": envelope_width,
        "balanced_envelope": _tensor_fingerprint(
            envelope,
            "balanced_envelope",
            chunk_bytes=config.fingerprint_chunk_bytes,
        ),
        "reduced_activation_rows": _tensor_fingerprint(
            activations,
            "reduced_activation_rows",
            chunk_bytes=config.fingerprint_chunk_bytes,
        ),
        "reduced_sensitivity_rows": _tensor_fingerprint(
            sensitivities,
            "reduced_sensitivity_rows",
            chunk_bytes=config.fingerprint_chunk_bytes,
        ),
        "candidate_manifest": candidate_manifest,
        "candidate_manifest_sha256": _canonical_sha256(candidate_manifest),
        "candidate_order": list(_CANDIDATE_ORDER),
        "equal_rank_random_control": True,
        "all_candidates_projected_out_of_inherited_right_row_space": True,
        "all_candidates_row_orthonormal": True,
        "role_local_input_vjp_formula": (
            "D@W_base + (D@L_inherited)@R_inherited"
        ),
        "role_local_input_vjp_precision": source_payload.get(
            "role_local_input_vjp_precision",
            "uniform-source-dtype",
        ),
        "full_width_replay_table_stored": False,
        "internal_weight_matching_used": False,
        "activation_target_matching_used": False,
        "optimizer_created": False,
        "determinism_verified_by_exact_repeat": True,
        "memory_plan_sha256": memory_plan_sha256,
    }
    report_payload["report_sha256"] = _canonical_sha256(report_payload)
    return GUDSiteBasisResult(
        output_jacobian_right_rows=jacobian_rows,
        balanced_right_rows=balanced_rows,
        procrustes_hybrid_right_rows=hybrid_rows,
        random_control_right_rows=random_rows,
        report=GUDSiteAuditReport(MappingProxyType(dict(report_payload))),
    )


def audit_gud_site_basis(
    site: GUDSiteAuditInput,
    role_spec: GUDRoleSpec,
    config: GUDOutputJacobianAuditConfig,
) -> GUDSiteBasisResult:
    """Build four authenticated candidates for one generic role-local site."""

    if not isinstance(site, GUDSiteAuditInput):
        raise GUDOutputJacobianAuditShapeError("site has the wrong type.")
    if not isinstance(role_spec, GUDRoleSpec):
        raise GUDOutputJacobianAuditShapeError("role_spec has the wrong type.")
    if not isinstance(config, GUDOutputJacobianAuditConfig):
        raise GUDOutputJacobianAuditShapeError("config has the wrong type.")
    geometry = GUDStageGeometry(
        logical_layers=1,
        hidden_size=(
            role_spec.input_width
            if role_spec.role in {"gate", "up"}
            else role_spec.output_width
        ),
        intermediate_size=(
            role_spec.output_width
            if role_spec.role in {"gate", "up"}
            else role_spec.input_width
        ),
    )
    memory_plan = plan_gud_auxiliary_memory(geometry, (site,), config)
    return _build_site(site, role_spec, config, memory_plan)


def audit_gud_stage_bases(
    geometry: GUDStageGeometry,
    sites: Sequence[GUDSiteAuditInput],
    config: GUDOutputJacobianAuditConfig,
    *,
    require_qwen_30_layer_geometry: bool = False,
) -> GUDStageAuditResult:
    """Audit a complete layer-major Gate/Up/Down stage sequentially."""

    if not isinstance(require_qwen_30_layer_geometry, bool):
        raise GUDOutputJacobianAuditShapeError(
            "require_qwen_30_layer_geometry must be boolean."
        )
    if require_qwen_30_layer_geometry:
        validate_qwen_30_layer_stage_geometry(geometry)
    site_values = tuple(sites)
    expected_keys = tuple(
        (layer, role)
        for layer in range(geometry.logical_layers)
        for role in GUD_ROLE_ORDER
    )
    actual_keys = tuple((site.layer_index, site.role) for site in site_values)
    if actual_keys != expected_keys:
        raise GUDOutputJacobianAuditShapeError(
            "Sites must be complete and ordered layer-major Gate, Up, Down."
        )
    memory_plan = plan_gud_auxiliary_memory(geometry, site_values, config)
    results = tuple(
        _build_site(site, geometry.role_spec(site.role), config, memory_plan)
        for site in site_values
    )
    basis_manifest = [
        {
            "layer_index": site.layer_index,
            "role": site.role,
            "site_report_sha256": result.report.report_sha256,
            "candidate_manifest_sha256": result.report.payload[
                "candidate_manifest_sha256"
            ],
        }
        for site, result in zip(site_values, results, strict=True)
    ]
    report_payload: dict[str, Any] = {
        "schema": f"{GUD_OUTPUT_JACOBIAN_AUDIT_SCHEMA}:stage-v1",
        "geometry": geometry.to_dict(),
        "geometry_sha256": _canonical_sha256(geometry.to_dict()),
        "qwen_30_layer_geometry_required": require_qwen_30_layer_geometry,
        "qwen_30_layer_geometry_validated": require_qwen_30_layer_geometry,
        "ordered_roles": list(GUD_ROLE_ORDER),
        "site_count": len(results),
        "basis_manifest": basis_manifest,
        "basis_manifest_sha256": _canonical_sha256(basis_manifest),
        "memory_plan": memory_plan,
        "memory_plan_sha256": memory_plan["memory_plan_sha256"],
        "sequential_site_processing": True,
        "separable_full_element_fingerprints": True,
        "full_lanes_rows_width_fingerprint_table_used": False,
        "internal_weight_matching_used": False,
        "determinism_verified": True,
    }
    report_payload["report_sha256"] = _canonical_sha256(report_payload)
    return GUDStageAuditResult(
        sites=results,
        report=GUDStageAuditReport(MappingProxyType(dict(report_payload))),
    )


__all__ = [
    "DEFAULT_FINGERPRINT_CHUNK_BYTES",
    "GUD_CANDIDATE_BASIS_SCHEMA",
    "GUD_OUTPUT_JACOBIAN_AUDIT_SCHEMA",
    "GUD_ROLE_ORDER",
    "HARD_MAX_AUXILIARY_BYTES",
    "QWEN_STAGE_HIDDEN_SIZE",
    "QWEN_STAGE_INTERMEDIATE_SIZE",
    "QWEN_STAGE_LOGICAL_LAYERS",
    "GUDJacobianBlock",
    "GUDOutputJacobianAuditConfig",
    "GUDOutputJacobianAuditDeterminismError",
    "GUDOutputJacobianAuditError",
    "GUDOutputJacobianAuditFiniteError",
    "GUDOutputJacobianAuditLimitError",
    "GUDOutputJacobianAuditRankError",
    "GUDOutputJacobianAuditShapeError",
    "GUDRoleSpec",
    "GUDSiteAuditInput",
    "GUDSiteAuditReport",
    "GUDSiteBasisResult",
    "GUDStageAuditReport",
    "GUDStageAuditResult",
    "GUDStageGeometry",
    "audit_gud_site_basis",
    "audit_gud_stage_bases",
    "finalize_streamed_gud_site_basis",
    "finalize_streaming_gud_envelope",
    "ordered_gud_role_specs",
    "plan_gud_auxiliary_memory",
    "role_local_input_vjp",
    "role_local_input_vjp_prevalidated",
    "validate_qwen_30_layer_stage_geometry",
]
