"""Transactional per-assignment Q/K/V/O role-scale edits.

Attention codebook merges may introduce least-squares role scales even when
the parent registry had none.  A low raw-weight dot product can drive V and O
scales almost to zero, muting the removed site's value cache and residual
write.  These scale parameters already exist in the compact checkpoint, so
learning better values changes neither architecture nor parameter count.

This module installs explicit scale candidates without comparing them to
parent weights.  It preserves every non-target tensor and parameter object,
supports exact rollback, and is intended to be paired with output-function
selection rather than tensor-similarity gates.
"""

from __future__ import annotations

import hashlib
import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from neural_forge.hf_attention_codebook import AttentionCodebookRegistry


ROLE_SCALE_EDIT_SCHEMA = "neural-forge-residual-attention-role-scale-edit-v2"
ROLE_NAMES = ("q", "k", "v", "o")


class ResidualAttentionRoleScaleError(RuntimeError):
    """A role-scale request, installation, or rollback is invalid."""


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ResidualAttentionRoleScaleError(f"{name} cannot be negative.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ResidualAttentionRoleScaleError(f"{name} cannot be negative.") from error
    if result < 0:
        raise ResidualAttentionRoleScaleError(f"{name} cannot be negative.")
    return result


def _finite_float(value: Any, name: str, *, maximum: float) -> float:
    if isinstance(value, bool):
        raise ResidualAttentionRoleScaleError(f"{name} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ResidualAttentionRoleScaleError(f"{name} must be finite.") from error
    if not math.isfinite(result) or abs(result) > maximum:
        raise ResidualAttentionRoleScaleError(
            f"{name} must be finite with absolute value <= {maximum}."
        )
    return result


def _tensor_sha256(value: Tensor | None) -> str | None:
    if value is None:
        return None
    detached = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(detached.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(detached.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _registry_hashes(
    registry: AttentionCodebookRegistry, *, exclude_role_scales: bool
) -> dict[str, str]:
    result = {}
    for name, value in sorted(registry.state_dict().items()):
        if exclude_role_scales and name == "role_scales":
            continue
        digest = _tensor_sha256(value)
        assert digest is not None
        result[name] = digest
    return result


def _registry_sha256(registry: AttentionCodebookRegistry) -> str:
    digest = hashlib.sha256()
    for name, value_hash in _registry_hashes(
        registry, exclude_role_scales=False
    ).items():
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value_hash.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class AttentionRoleScales:
    q: float
    k: float
    v: float
    o: float

    @classmethod
    def validated(
        cls,
        values: Mapping[str, Any] | AttentionRoleScales,
        *,
        max_abs_scale: float = 16.0,
    ) -> AttentionRoleScales:
        maximum = _finite_float(max_abs_scale, "max_abs_scale", maximum=float("inf"))
        if maximum <= 0.0:
            raise ResidualAttentionRoleScaleError("max_abs_scale must be positive.")
        if isinstance(values, cls):
            raw = {name: getattr(values, name) for name in ROLE_NAMES}
        elif isinstance(values, Mapping) and set(values) == set(ROLE_NAMES):
            raw = dict(values)
        else:
            raise ResidualAttentionRoleScaleError(
                "Role scales must provide exactly q, k, v, and o."
            )
        return cls(
            **{
                name: _finite_float(raw[name], f"{name}_scale", maximum=maximum)
                for name in ROLE_NAMES
            }
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.q, self.k, self.v, self.o

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in ROLE_NAMES}


@dataclass(frozen=True, slots=True)
class RoleScaleInstallationReport:
    layer_index: int
    physical_group: int
    canonical_group: int
    before_scales: AttentionRoleScales
    requested_scales: AttentionRoleScales
    installed_scales: AttentionRoleScales
    role_scales_dtype: str
    role_scales_device: str
    role_scales_sha256_before: str
    role_scales_sha256_installed: str
    registry_sha256_before: str
    registry_sha256_installed: str
    non_scale_state_sha256: str
    parameter_identity_sha256: str
    materialized_projection_sha256s: tuple[tuple[str, str, str | None], ...]
    changed_scalar_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ROLE_SCALE_EDIT_SCHEMA,
            "layer_index": self.layer_index,
            "physical_group": self.physical_group,
            "canonical_group": self.canonical_group,
            "before_scales": self.before_scales.as_dict(),
            "requested_scales": self.requested_scales.as_dict(),
            "installed_scales": self.installed_scales.as_dict(),
            "role_scales_dtype": self.role_scales_dtype,
            "role_scales_device": self.role_scales_device,
            "requested_values_exactly_representable": (
                self.requested_scales == self.installed_scales
            ),
            "role_scales_sha256_before": self.role_scales_sha256_before,
            "role_scales_sha256_installed": self.role_scales_sha256_installed,
            "registry_sha256_before": self.registry_sha256_before,
            "registry_sha256_installed": self.registry_sha256_installed,
            "non_scale_state_sha256": self.non_scale_state_sha256,
            "parameter_identity_sha256": self.parameter_identity_sha256,
            "materialized_projection_sha256s": {
                role: {"weight": weight, "bias": bias}
                for role, weight, bias in self.materialized_projection_sha256s
            },
            "changed_scalar_count": self.changed_scalar_count,
            "runtime_parameters_added": 0,
            "parameter_count_unchanged": True,
            "weights_required_to_match_parent": False,
            "selection_contract": "output-function-only",
            "optimizer_steps": 0,
            "backward_calls": 0,
        }


def _parameter_identity_sha256(registry: AttentionCodebookRegistry) -> str:
    return hashlib.sha256(
        repr(
            tuple(
                (
                    name,
                    id(value),
                    tuple(value.shape),
                    str(value.dtype),
                    str(value.device),
                    bool(value.requires_grad),
                )
                for name, value in registry.named_parameters()
            )
        ).encode("utf-8")
    ).hexdigest()


def _mapping(
    registry: AttentionCodebookRegistry,
    layer_index: int,
    physical_group: int,
) -> tuple[int, int]:
    layer = _nonnegative_int(layer_index, "layer_index")
    physical = _nonnegative_int(physical_group, "physical_group")
    if layer >= int(registry.logical_layers) or physical >= int(
        registry.key_value_heads
    ):
        raise ResidualAttentionRoleScaleError(
            "Role-scale layer or physical group is out of range."
        )
    raw_permutation = registry.group_permutations[layer].detach().cpu()
    if raw_permutation.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ResidualAttentionRoleScaleError(
            "group_permutations row must contain integers."
        )
    permutation = raw_permutation.to(torch.long)
    expected = torch.arange(int(registry.key_value_heads), dtype=torch.long)
    if tuple(permutation.shape) != (int(registry.key_value_heads),) or not torch.equal(
        torch.sort(permutation).values, expected
    ):
        raise ResidualAttentionRoleScaleError(
            "group_permutations row must be a complete physical-group bijection."
        )
    matches = torch.nonzero(permutation == physical, as_tuple=False).reshape(-1)
    if int(matches.numel()) != 1:
        raise ResidualAttentionRoleScaleError(
            "group_permutations does not map the physical group bijectively."
        )
    return layer, int(matches.item())


def _row_scales(value: Tensor) -> AttentionRoleScales:
    if tuple(value.shape) != (4,) or not bool(torch.isfinite(value.detach()).all()):
        raise ResidualAttentionRoleScaleError(
            "One assignment role-scale row must contain four finite values."
        )
    return AttentionRoleScales(*map(float, value.detach().cpu().tolist()))


def _materialized_hashes(
    registry: AttentionCodebookRegistry, layer_index: int
) -> tuple[tuple[str, str, str | None], ...]:
    result = []
    for role in ROLE_NAMES:
        weight, bias = registry.materialize_projection(layer_index, role)
        weight_hash = _tensor_sha256(weight)
        assert weight_hash is not None
        result.append((role, weight_hash, _tensor_sha256(bias)))
    return tuple(result)


class InstalledAttentionRoleScales:
    """One applied assignment-scale edit with exact rollback until committed."""

    def __init__(
        self,
        registry: AttentionCodebookRegistry,
        report: RoleScaleInstallationReport,
        before_row: Tensor,
        before_full_scales: Tensor,
        before_non_scale_hashes: Mapping[str, str],
    ) -> None:
        self.registry = registry
        self.report = report
        self._before_row = before_row
        self._before_full_scales = before_full_scales
        self._before_non_scale_hashes = dict(before_non_scale_hashes)
        self._rolled_back = False
        self._committed = False

    @property
    def is_applied(self) -> bool:
        return not self._rolled_back

    @property
    def is_committed(self) -> bool:
        return self._committed

    def assert_live(self) -> RoleScaleInstallationReport:
        if self._rolled_back:
            raise ResidualAttentionRoleScaleError(
                "A rolled-back role-scale edit is no longer live."
            )
        scales = self.registry.role_scales
        if scales is None:
            raise ResidualAttentionRoleScaleError("Live registry lost role scales.")
        row = scales[self.report.layer_index, self.report.canonical_group]
        if (
            _row_scales(row) != self.report.installed_scales
            or _tensor_sha256(scales) != self.report.role_scales_sha256_installed
            or _registry_sha256(self.registry) != self.report.registry_sha256_installed
            or _registry_hashes(self.registry, exclude_role_scales=True)
            != self._before_non_scale_hashes
            or _parameter_identity_sha256(self.registry)
            != self.report.parameter_identity_sha256
            or _materialized_hashes(self.registry, self.report.layer_index)
            != self.report.materialized_projection_sha256s
        ):
            raise ResidualAttentionRoleScaleError(
                "Live registry no longer matches the installed role-scale edit."
            )
        return self.report

    def commit(self) -> RoleScaleInstallationReport:
        if self._rolled_back:
            raise ResidualAttentionRoleScaleError(
                "A rolled-back role-scale edit cannot be committed."
            )
        self.assert_live()
        self._committed = True
        return self.report

    def rollback(self) -> None:
        if self._committed:
            raise ResidualAttentionRoleScaleError(
                "A committed role-scale edit cannot be rolled back by its handle."
            )
        if self._rolled_back:
            return
        self.assert_live()
        scales = self.registry.role_scales
        assert scales is not None
        with torch.no_grad():
            scales[self.report.layer_index, self.report.canonical_group].copy_(
                self._before_row
            )
        exact = (
            bool(torch.equal(scales.detach(), self._before_full_scales))
            and _tensor_sha256(scales) == self.report.role_scales_sha256_before
            and _registry_sha256(self.registry) == self.report.registry_sha256_before
            and _registry_hashes(self.registry, exclude_role_scales=True)
            == self._before_non_scale_hashes
            and _parameter_identity_sha256(self.registry)
            == self.report.parameter_identity_sha256
        )
        if exact:
            self._rolled_back = True
        if not exact:
            raise ResidualAttentionRoleScaleError(
                "Exact role-scale rollback identity audit failed."
            )

    def __enter__(self) -> InstalledAttentionRoleScales:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        if not self._committed:
            self.rollback()


def install_attention_role_scales(
    registry: AttentionCodebookRegistry,
    *,
    layer_index: int,
    physical_group: int,
    scales: Mapping[str, Any] | AttentionRoleScales,
    max_abs_scale: float = 16.0,
) -> InstalledAttentionRoleScales:
    """Install one explicit physical-site scale row without changing weights."""

    if not isinstance(registry, AttentionCodebookRegistry):
        raise ResidualAttentionRoleScaleError(
            "registry must be an AttentionCodebookRegistry."
        )
    role_scales = registry.role_scales
    if role_scales is None or tuple(role_scales.shape) != (
        registry.logical_layers,
        registry.key_value_heads,
        4,
    ):
        raise ResidualAttentionRoleScaleError(
            "Role-scale search requires an existing [layers,groups,4] parameter."
        )
    layer, canonical = _mapping(registry, layer_index, physical_group)
    installed = AttentionRoleScales.validated(scales, max_abs_scale=max_abs_scale)
    before_full = role_scales.detach().clone()
    before_row = before_full[layer, canonical].clone()
    before = _row_scales(before_row)
    before_scale_hash = _tensor_sha256(role_scales)
    assert before_scale_hash is not None
    before_registry_hash = _registry_sha256(registry)
    non_scale_hashes = _registry_hashes(registry, exclude_role_scales=True)
    non_scale_sha = hashlib.sha256(repr(non_scale_hashes).encode("utf-8")).hexdigest()
    parameter_identity = _parameter_identity_sha256(registry)
    try:
        with torch.no_grad():
            role_scales[layer, canonical].copy_(
                torch.tensor(
                    installed.as_tuple(),
                    device=role_scales.device,
                    dtype=role_scales.dtype,
                )
            )
        installed_actual = _row_scales(role_scales[layer, canonical])
        installed_scale_hash = _tensor_sha256(role_scales)
        assert installed_scale_hash is not None
        if (
            _registry_hashes(registry, exclude_role_scales=True) != non_scale_hashes
            or _parameter_identity_sha256(registry) != parameter_identity
        ):
            raise ResidualAttentionRoleScaleError(
                "Role-scale installation changed unrelated state or parameter identity."
            )
        changed = int(
            torch.count_nonzero(
                before_row != role_scales[layer, canonical].detach()
            ).item()
        )
        report = RoleScaleInstallationReport(
            layer_index=layer,
            physical_group=_nonnegative_int(physical_group, "physical_group"),
            canonical_group=canonical,
            before_scales=before,
            requested_scales=installed,
            installed_scales=installed_actual,
            role_scales_dtype=str(role_scales.dtype),
            role_scales_device=str(role_scales.device),
            role_scales_sha256_before=before_scale_hash,
            role_scales_sha256_installed=installed_scale_hash,
            registry_sha256_before=before_registry_hash,
            registry_sha256_installed=_registry_sha256(registry),
            non_scale_state_sha256=non_scale_sha,
            parameter_identity_sha256=parameter_identity,
            materialized_projection_sha256s=_materialized_hashes(registry, layer),
            changed_scalar_count=changed,
        )
        handle = InstalledAttentionRoleScales(
            registry,
            report,
            before_row,
            before_full,
            non_scale_hashes,
        )
        handle.assert_live()
        return handle
    except Exception:
        with torch.no_grad():
            role_scales.copy_(before_full)
        raise


__all__ = [
    "ROLE_NAMES",
    "ROLE_SCALE_EDIT_SCHEMA",
    "AttentionRoleScales",
    "InstalledAttentionRoleScales",
    "ResidualAttentionRoleScaleError",
    "RoleScaleInstallationReport",
    "install_attention_role_scales",
]
