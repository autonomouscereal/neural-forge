"""Authenticated role-specific site masks for guided Q/K/V/O training.

The mask is derived only from the already-passed output-Jacobian audit.  For
each projection role it retains the smallest energy-ordered site prefix that
captures at least eighty percent of eligible normal energy, with a hard floor
of eight sites.  ``GradientSiteMaskGuard`` is a trusted in-process training
guard: it masks row-zero gradients and proves that every unselected output
factor remains exact zero while every unselected input factor remains the
authenticated audit basis.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from neural_forge.hf_dense_gqa_qkvo_residual_adapter import (
    QKVO_OUTPUT_JACOBIAN_AUDIT_V2_SCHEMA,
    GuidedSparseDenseGQAQKVOResidualRegistry,
)
from neural_forge.hf_output_jacobian_sketch import OUTPUT_JACOBIAN_SKETCH_SCHEMA


ProjectionRole = Literal["q", "k", "v", "o"]
GUIDED_ATTENTION_SITE_MASK_SCHEMA = (
    "neural-forge-guided-attention-site-mask-v1"
)
GUIDED_ATTENTION_ROLE_MASK_SCHEMA = (
    "neural-forge-guided-attention-role-site-mask-v1"
)
NORMAL_ENERGY_COVERAGE = 0.80
MINIMUM_SELECTED_SITES = 8
EXACT_SITE_COUNT = 120
EXACT_SKETCH_COUNT = 480
ROLES: tuple[ProjectionRole, ...] = ("q", "k", "v", "o")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROLE_AUTHORIZATION_POLICY = (
    "installers-must-ignore-all-tensors-for-roles-not-listed-in-"
    "authorized_roles"
)


class GuidedAttentionSiteMaskError(RuntimeError):
    """An audit, mask, topology, or guarded-state invariant failed."""


class GuidedAttentionSiteMaskAuthenticationError(GuidedAttentionSiteMaskError):
    """The source audit or one of its nested reports did not authenticate."""


class GradientSiteMaskGuardError(GuidedAttentionSiteMaskError):
    """A live guided registry violated its role-specific site mask."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise GuidedAttentionSiteMaskAuthenticationError(
            "Value has no finite canonical JSON representation."
        ) from error


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise GuidedAttentionSiteMaskAuthenticationError(
            f"{name} must be a lowercase SHA-256."
        )
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise GuidedAttentionSiteMaskAuthenticationError(
            f"{name} must be boolean."
        )
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GuidedAttentionSiteMaskAuthenticationError(
            f"{name} must be a nonnegative integer."
        )
    return value


def _finite_float(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GuidedAttentionSiteMaskAuthenticationError(
            f"{name} must be numeric."
        )
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise GuidedAttentionSiteMaskAuthenticationError(
            f"{name} must be finite and at least {minimum}."
        )
    return result


def _report_without_self_hash(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    payload = dict(value)
    supplied = _require_sha256(payload.pop("report_sha256", None), f"{name}.report_sha256")
    if _canonical_sha256(payload) != supplied:
        raise GuidedAttentionSiteMaskAuthenticationError(
            f"{name} canonical SHA-256 does not verify."
        )
    return payload


def _tensor_sha256(value: Tensor) -> str:
    detached = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(
        json.dumps(list(detached.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(memoryview(detached.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def _exact_zero_bytes(value: Tensor) -> bool:
    detached = value.detach().cpu().contiguous()
    raw = detached.reshape(-1).view(torch.uint8).numpy().tobytes()
    return raw == bytes(len(raw))


@dataclass(frozen=True, slots=True)
class GuidedAttentionRoleSiteMask:
    """One role's authenticated normal-energy prefix."""

    role: ProjectionRole
    eligible_site_energies: tuple[tuple[int, float], ...]
    selected_site_rows: tuple[int, ...]
    eligible_normal_energy: float
    selected_normal_energy: float
    selected_energy_coverage: float
    eligible_site_count: int
    selected_site_count: int

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise GuidedAttentionSiteMaskError("Role mask must be Q, K, V, or O.")
        rows = tuple(self.eligible_site_energies)
        if len(rows) < MINIMUM_SELECTED_SITES:
            raise GuidedAttentionSiteMaskError(
                "Each role needs at least eight eligible audit sites."
            )
        normalized: list[tuple[int, float]] = []
        for index, row in enumerate(rows):
            if not isinstance(row, tuple) or len(row) != 2:
                raise GuidedAttentionSiteMaskError(
                    f"eligible_site_energies[{index}] is malformed."
                )
            site = _nonnegative_int(row[0], f"eligible site {index}")
            energy = _finite_float(row[1], f"eligible energy {index}")
            if site >= EXACT_SITE_COUNT or energy <= 0.0:
                raise GuidedAttentionSiteMaskError(
                    "Eligible sites require an in-range row and positive energy."
                )
            normalized.append((site, energy))
        expected_order = sorted(normalized, key=lambda item: (-item[1], item[0]))
        if normalized != expected_order or len({row for row, _ in normalized}) != len(
            normalized
        ):
            raise GuidedAttentionSiteMaskError(
                "Eligible sites must be unique and ordered by energy then row."
            )
        selected = tuple(self.selected_site_rows)
        if selected != tuple(row for row, _energy in normalized[: len(selected)]):
            raise GuidedAttentionSiteMaskError(
                "Selected sites must be an exact prefix of eligible energy order."
            )
        if len(selected) < MINIMUM_SELECTED_SITES:
            raise GuidedAttentionSiteMaskError(
                "Selected site prefix is smaller than the eight-site floor."
            )
        eligible_energy = math.fsum(energy for _row, energy in normalized)
        selected_energy = math.fsum(
            energy for _row, energy in normalized[: len(selected)]
        )
        coverage = selected_energy / eligible_energy
        if (
            self.eligible_site_count != len(normalized)
            or self.selected_site_count != len(selected)
            or not math.isclose(
                float(self.eligible_normal_energy),
                eligible_energy,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(self.selected_normal_energy),
                selected_energy,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(self.selected_energy_coverage),
                coverage,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or coverage < NORMAL_ENERGY_COVERAGE
        ):
            raise GuidedAttentionSiteMaskError(
                "Role-mask energy accounting or coverage is invalid."
            )
        previous_count = len(selected) - 1
        if previous_count >= MINIMUM_SELECTED_SITES:
            previous = math.fsum(
                energy for _row, energy in normalized[:previous_count]
            )
            if previous / eligible_energy >= NORMAL_ENERGY_COVERAGE:
                raise GuidedAttentionSiteMaskError(
                    "Selected sites are not the minimal admissible prefix."
                )
        object.__setattr__(self, "eligible_site_energies", tuple(normalized))
        object.__setattr__(self, "selected_site_rows", selected)
        object.__setattr__(self, "eligible_normal_energy", eligible_energy)
        object.__setattr__(self, "selected_normal_energy", selected_energy)
        object.__setattr__(self, "selected_energy_coverage", coverage)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": GUIDED_ATTENTION_ROLE_MASK_SCHEMA,
            "role": self.role,
            "eligible_site_energies": [
                {"site_row": row, "normal_frobenius_sq_estimate": energy}
                for row, energy in self.eligible_site_energies
            ],
            "selected_site_rows": list(self.selected_site_rows),
            "eligible_normal_energy": self.eligible_normal_energy,
            "selected_normal_energy": self.selected_normal_energy,
            "selected_energy_coverage": self.selected_energy_coverage,
            "eligible_site_count": self.eligible_site_count,
            "selected_site_count": self.selected_site_count,
            "minimum_energy_coverage": NORMAL_ENERGY_COVERAGE,
            "minimum_selected_sites": MINIMUM_SELECTED_SITES,
            "maximum_selected_sites": None,
            "selection_order": "descending-normal-energy-then-site-row",
        }


@dataclass(frozen=True, slots=True)
class GuidedAttentionSiteMask:
    """Immutable role-specific mask bound to one exact QKVO audit."""

    audit_report_sha256: str
    site_indices: tuple[tuple[int, int], ...]
    site_indices_sha256: str
    role_masks: tuple[GuidedAttentionRoleSiteMask, ...]
    mask_contract_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        audit_sha = _require_sha256(
            self.audit_report_sha256, "audit_report_sha256"
        )
        sites = tuple(tuple(value) for value in self.site_indices)
        expected_sites = tuple(
            (layer, group) for layer in range(30) for group in range(4)
        )
        if sites != expected_sites:
            raise GuidedAttentionSiteMaskError(
                "Mask requires the exact canonical 30x4 site index."
            )
        site_sha = _require_sha256(
            self.site_indices_sha256, "site_indices_sha256"
        )
        if _canonical_sha256([list(value) for value in sites]) != site_sha:
            raise GuidedAttentionSiteMaskError("Mask site-index SHA does not verify.")
        roles = tuple(self.role_masks)
        if tuple(value.role for value in roles) != ROLES:
            raise GuidedAttentionSiteMaskError(
                "Mask must contain ordered Q, K, V, and O role contracts."
            )
        object.__setattr__(self, "audit_report_sha256", audit_sha)
        object.__setattr__(self, "site_indices", sites)
        object.__setattr__(self, "site_indices_sha256", site_sha)
        object.__setattr__(self, "role_masks", roles)
        object.__setattr__(
            self, "mask_contract_sha256", _canonical_sha256(self._payload())
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": GUIDED_ATTENTION_SITE_MASK_SCHEMA,
            "audit_report_sha256": self.audit_report_sha256,
            "site_indices": [list(value) for value in self.site_indices],
            "site_indices_sha256": self.site_indices_sha256,
            "normal_energy_coverage": NORMAL_ENERGY_COVERAGE,
            "minimum_selected_sites_per_role": MINIMUM_SELECTED_SITES,
            "maximum_selected_sites_per_role": None,
            "eligibility": {
                "sketch_gate_passed": True,
                "minimum_normal_effective_rank": 8,
                "positive_normal_frobenius_sq_estimate": True,
                "block_identity_replay_authenticated": True,
            },
            "roles": {value.role: value.as_dict() for value in self.role_masks},
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload(), "mask_contract_sha256": self.mask_contract_sha256}

    def for_role(self, role: ProjectionRole) -> GuidedAttentionRoleSiteMask:
        if role not in ROLES:
            raise GuidedAttentionSiteMaskError("Unknown QKVO role.")
        return self.role_masks[ROLES.index(role)]

    @property
    def selected_rows_by_role(self) -> Mapping[ProjectionRole, tuple[int, ...]]:
        return MappingProxyType(
            {value.role: value.selected_site_rows for value in self.role_masks}
        )


def _authenticate_audit(
    audit_report: Mapping[str, Any] | Any,
    *,
    expected_report_sha256: str,
) -> tuple[dict[str, Any], tuple[tuple[int, int], ...]]:
    if callable(getattr(audit_report, "to_dict", None)):
        audit_report = audit_report.to_dict()
    if not isinstance(audit_report, Mapping):
        raise GuidedAttentionSiteMaskAuthenticationError(
            "audit_report must be a mapping or canonical report object."
        )
    report = dict(audit_report)
    expected = _require_sha256(expected_report_sha256, "expected_report_sha256")
    declared = _require_sha256(report.get("report_sha256"), "report.report_sha256")
    if declared != expected:
        raise GuidedAttentionSiteMaskAuthenticationError(
            "Audit report does not match its external SHA-256 root."
        )
    _report_without_self_hash(report, "audit_report")
    if (
        report.get("schema") != QKVO_OUTPUT_JACOBIAN_AUDIT_V2_SCHEMA
        or report.get("passed") is not True
        or report.get("site_count") != EXACT_SITE_COUNT
        or report.get("role_count") != len(ROLES)
        or report.get("sketch_count") != EXACT_SKETCH_COUNT
    ):
        raise GuidedAttentionSiteMaskAuthenticationError(
            "Audit schema, decision, or exact topology changed."
        )
    config = report.get("config")
    if not isinstance(config, Mapping) or (
        config.get("prompt_count"),
        config.get("horizon"),
        config.get("rank"),
        tuple(config.get("lanes", ())),
    ) != (24, 8, 8, ("parent-greedy", "child-greedy")):
        raise GuidedAttentionSiteMaskAuthenticationError(
            "Audit configuration is not the exact 24x8 rank-eight replay."
        )
    if report.get("config_sha256") != _canonical_sha256(dict(config)):
        raise GuidedAttentionSiteMaskAuthenticationError(
            "Audit configuration SHA-256 does not verify."
        )

    sites_value = report.get("site_indices")
    if not isinstance(sites_value, list):
        raise GuidedAttentionSiteMaskAuthenticationError(
            "Audit site_indices must be a JSON list."
        )
    try:
        sites = tuple(
            (_nonnegative_int(row[0], "site layer"), _nonnegative_int(row[1], "site group"))
            for row in sites_value
            if isinstance(row, list) and len(row) == 2
        )
    except (IndexError, TypeError) as error:
        raise GuidedAttentionSiteMaskAuthenticationError(
            "Audit site_indices are malformed."
        ) from error
    expected_sites = tuple(
        (layer, group) for layer in range(30) for group in range(4)
    )
    if sites != expected_sites or len(sites) != len(sites_value):
        raise GuidedAttentionSiteMaskAuthenticationError(
            "Audit site indices are not the exact canonical 30x4 order."
        )
    site_sha = _require_sha256(
        report.get("site_indices_sha256"), "site_indices_sha256"
    )
    if _canonical_sha256(sites_value) != site_sha:
        raise GuidedAttentionSiteMaskAuthenticationError(
            "Audit site-index SHA-256 does not verify."
        )

    sha_pairs = (
        ("pass1_replay_sha256", "pass2_replay_sha256"),
        (
            "pass1_full_element_replay_authenticator_sha256",
            "pass2_full_element_replay_authenticator_sha256",
        ),
        ("native_state_sha256_before", "native_state_sha256_after"),
        ("factor_state_sha256_before", "factor_state_sha256_after"),
        (
            "inherited_qkvo_state_sha256_before",
            "inherited_qkvo_state_sha256_after",
        ),
        (
            "factor_parameter_versions_sha256_before",
            "factor_parameter_versions_sha256_after",
        ),
    )
    for before_name, after_name in sha_pairs:
        before = _require_sha256(report.get(before_name), before_name)
        after = _require_sha256(report.get(after_name), after_name)
        if before != after:
            raise GuidedAttentionSiteMaskAuthenticationError(
                f"Audit before/after evidence changed for {before_name}."
            )
    required_true = (
        "probabilistic_replay_authenticator_match",
        "replay_authentication_binds_every_element",
        "factor_gradients_cleared",
        "hooks_removed",
        "transaction_live_after",
        "rollback_compatible_after",
        "installer_must_ignore_unauthorized_role_tensors",
        "all_four_role_tensors_retained_for_diagnostics",
    )
    if any(_strict_bool(report.get(name), name) is not True for name in required_true):
        raise GuidedAttentionSiteMaskAuthenticationError(
            "Audit replay, mutation, lifecycle, or installer authentication failed."
        )
    active_hook_count = _nonnegative_int(
        report.get("active_hook_count_after"), "active_hook_count_after"
    )
    projection_lanes = _nonnegative_int(
        report.get("replay_authentication_projection_lanes"),
        "replay_authentication_projection_lanes",
    )
    if (
        active_hook_count != 0
        or report.get("optimizer_created") is not False
        or report.get("optimizer_used") is not False
        or report.get("optimizer_steps") != 0
        or report.get("new_candidate_accessed") is not False
        or report.get("sealed_accessed") is not False
        or report.get("replay_authentication_is_collision_free") is not False
        or projection_lanes <= 0
    ):
        raise GuidedAttentionSiteMaskAuthenticationError(
            "Audit execution/replay attestations are inconsistent."
        )
    aggregate = report.get("aggregate")
    per_role = report.get("per_role")
    if (
        not isinstance(aggregate, Mapping)
        or aggregate.get("passed") is not True
        or not isinstance(per_role, Mapping)
        or set(per_role) != set(ROLES)
        or any(
            not isinstance(per_role[role], Mapping)
            or per_role[role].get("passed") is not True
            for role in ROLES
        )
        or report.get("diagnostic_passed_roles") != list(ROLES)
        or report.get("authorized_roles") != list(ROLES)
        or report.get("installable_roles") != list(ROLES)
        or report.get("role_authorization_policy") != _ROLE_AUTHORIZATION_POLICY
    ):
        raise GuidedAttentionSiteMaskAuthenticationError(
            "Audit role authorization is incomplete or malformed."
        )
    return report, sites


def build_guided_attention_site_mask(
    audit_report: Mapping[str, Any] | Any,
    *,
    expected_report_sha256: str,
) -> GuidedAttentionSiteMask:
    """Authenticate an exact audit and build four minimal 80%-energy prefixes."""

    report, sites = _authenticate_audit(
        audit_report, expected_report_sha256=expected_report_sha256
    )
    raw_site_reports = report.get("site_reports")
    if not isinstance(raw_site_reports, list) or len(raw_site_reports) != (
        EXACT_SKETCH_COUNT
    ):
        raise GuidedAttentionSiteMaskAuthenticationError(
            "Audit must contain exactly 480 site reports."
        )
    eligible: dict[ProjectionRole, list[tuple[int, float]]] = {
        role: [] for role in ROLES
    }
    for index, value in enumerate(raw_site_reports):
        if not isinstance(value, Mapping):
            raise GuidedAttentionSiteMaskAuthenticationError(
                f"site_reports[{index}] is not an object."
            )
        expected_role = ROLES[index // EXACT_SITE_COUNT]
        expected_row = index % EXACT_SITE_COUNT
        site_row = _nonnegative_int(
            value.get("site_row"), f"site_reports[{index}].site_row"
        )
        layer = _nonnegative_int(
            value.get("layer"), f"site_reports[{index}].layer"
        )
        group = _nonnegative_int(
            value.get("physical_group"),
            f"site_reports[{index}].physical_group",
        )
        if (
            value.get("role") != expected_role
            or site_row != expected_row
            or layer != sites[expected_row][0]
            or group != sites[expected_row][1]
        ):
            raise GuidedAttentionSiteMaskAuthenticationError(
                "Site reports are not in exact role/site-index order."
            )
        _finite_float(
            value.get("inherited_right_median_row_norm"),
            f"site_reports[{index}].inherited_right_median_row_norm",
        )
        sketch = value.get("sketch_report")
        if not isinstance(sketch, Mapping):
            raise GuidedAttentionSiteMaskAuthenticationError(
                f"site_reports[{index}].sketch_report is malformed."
            )
        if sketch.get("schema") != OUTPUT_JACOBIAN_SKETCH_SCHEMA:
            raise GuidedAttentionSiteMaskAuthenticationError(
                "A nested sketch report has the wrong schema."
            )
        _report_without_self_hash(sketch, f"site_reports[{index}].sketch_report")
        gate = _strict_bool(
            sketch.get("gate_passed"),
            f"site_reports[{index}].sketch_report.gate_passed",
        )
        block_replay = _strict_bool(
            sketch.get("block_identity_replay_authenticated"),
            f"site_reports[{index}].sketch_report.block replay",
        )
        rank = _nonnegative_int(
            sketch.get("normal_effective_rank"),
            f"site_reports[{index}].sketch_report.normal_effective_rank",
        )
        energy = _finite_float(
            sketch.get("normal_frobenius_sq_estimate"),
            f"site_reports[{index}].sketch_report.normal energy",
        )
        if gate and rank >= 8 and energy > 0.0 and block_replay:
            eligible[expected_role].append((expected_row, energy))

    role_masks: list[GuidedAttentionRoleSiteMask] = []
    for role in ROLES:
        ordered = sorted(eligible[role], key=lambda item: (-item[1], item[0]))
        if len(ordered) < MINIMUM_SELECTED_SITES:
            raise GuidedAttentionSiteMaskError(
                f"Role {role.upper()} has fewer than eight eligible sites."
            )
        total = math.fsum(energy for _row, energy in ordered)
        selected_count = MINIMUM_SELECTED_SITES
        cumulative = 0.0
        for count, (_row, energy) in enumerate(ordered, start=1):
            cumulative = math.fsum((cumulative, energy))
            if count >= MINIMUM_SELECTED_SITES and (
                cumulative / total >= NORMAL_ENERGY_COVERAGE
            ):
                selected_count = count
                break
        else:  # pragma: no cover - positive finite energy always reaches one.
            selected_count = len(ordered)
        selected_energy = math.fsum(
            energy for _row, energy in ordered[:selected_count]
        )
        role_masks.append(
            GuidedAttentionRoleSiteMask(
                role=role,
                eligible_site_energies=tuple(ordered),
                selected_site_rows=tuple(
                    row for row, _energy in ordered[:selected_count]
                ),
                eligible_normal_energy=total,
                selected_normal_energy=selected_energy,
                selected_energy_coverage=selected_energy / total,
                eligible_site_count=len(ordered),
                selected_site_count=selected_count,
            )
        )
    return GuidedAttentionSiteMask(
        audit_report_sha256=_require_sha256(
            report.get("report_sha256"), "report.report_sha256"
        ),
        site_indices=sites,
        site_indices_sha256=_require_sha256(
            report.get("site_indices_sha256"), "site_indices_sha256"
        ),
        role_masks=tuple(role_masks),
    )


class GradientSiteMaskGuard:
    """Trusted context guard for one full guided rank-eight registry.

    The callback hooks mask gradients only.  ``verify_step`` is deliberately
    explicit so callers check immediately after every optimizer step and catch
    weight decay, stale optimizer moments, or direct mutation.
    """

    def __init__(
        self,
        registry: GuidedSparseDenseGQAQKVOResidualRegistry,
        mask: GuidedAttentionSiteMask,
    ) -> None:
        if not isinstance(registry, GuidedSparseDenseGQAQKVOResidualRegistry):
            raise GradientSiteMaskGuardError(
                "registry must be GuidedSparseDenseGQAQKVOResidualRegistry."
            )
        if not isinstance(mask, GuidedAttentionSiteMask):
            raise GradientSiteMaskGuardError(
                "mask must be an authenticated GuidedAttentionSiteMask."
            )
        self.registry = registry
        self.mask = mask
        self._handles: list[RemovableHandle] = []
        self._entered = False
        self._closed = False
        self._unselected_rows: dict[ProjectionRole, tuple[int, ...]] = {}
        self._unselected_left_sha256s: dict[ProjectionRole, str] = {}
        self._unselected_right_sha256s: dict[ProjectionRole, str] = {}
        self._parameter_signatures: dict[str, tuple[int, int, tuple[int, ...], str, str]] = {}
        self._site_indices_signature: tuple[int, int, int, str] | None = None

    @property
    def active_hook_count(self) -> int:
        return len(self._handles)

    def _validate_topology(self, *, source_basis_required: bool) -> None:
        registry = self.registry
        assertion = getattr(registry, "assert_live", None)
        if callable(assertion):
            assertion()
        sites = tuple(
            (int(layer), int(group))
            for layer, group in registry.site_indices.detach().cpu().tolist()
        )
        if (
            sites != self.mask.site_indices
            or _canonical_sha256([list(value) for value in sites])
            != self.mask.site_indices_sha256
            or registry.site_count != EXACT_SITE_COUNT
            or registry.rank != 8
            or getattr(registry, "audit_report_sha256", None)
            != self.mask.audit_report_sha256
        ):
            raise GradientSiteMaskGuardError(
                "Guided registry topology/audit binding differs from the mask."
            )
        if source_basis_required:
            source_hashes = getattr(registry, "source_basis_tensor_sha256s", None)
            if not isinstance(source_hashes, Mapping) or set(source_hashes) != set(
                ROLES
            ):
                raise GradientSiteMaskGuardError(
                    "Guided registry lacks four authenticated source-basis hashes."
                )
            for role in ROLES:
                if _tensor_sha256(getattr(registry, f"{role}_right")) != (
                    source_hashes[role]
                ):
                    raise GradientSiteMaskGuardError(
                        f"Guided {role.upper()} right factors are not the source basis."
                    )

    @staticmethod
    def _signature(value: nn.Parameter) -> tuple[int, int, tuple[int, ...], str, str]:
        return (
            id(value),
            int(value.data_ptr()),
            tuple(value.shape),
            str(value.dtype),
            str(value.device),
        )

    def _register_hook(
        self, parameter: nn.Parameter, unselected_rows: tuple[int, ...]
    ) -> RemovableHandle:
        rows = torch.tensor(unselected_rows, dtype=torch.long, device=parameter.device)

        def mask_gradient(gradient: Tensor) -> Tensor:
            if not isinstance(gradient, Tensor) or tuple(gradient.shape) != tuple(
                parameter.shape
            ):
                raise GradientSiteMaskGuardError(
                    "A factor gradient does not match its guarded parameter."
                )
            result = gradient.clone(memory_format=torch.preserve_format)
            result.index_fill_(0, rows, 0)
            return result

        # Production enters this guard during ``guided-left`` while every
        # guided-right factor is intentionally frozen.  PyTorch refuses
        # ``register_hook`` on a Tensor whose current ``requires_grad`` flag is
        # false, but a hook registered while a leaf Parameter is enabled
        # remains attached across a later false -> true stage transition.
        # Temporarily enable only for registration and restore the exact flag
        # before returning (including every failure path); the guard must not
        # itself advance the tranche's training stage.
        requires_grad = parameter.requires_grad
        if not requires_grad:
            parameter.requires_grad_(True)
        try:
            return parameter.register_hook(mask_gradient)
        finally:
            if parameter.requires_grad is not requires_grad:
                parameter.requires_grad_(requires_grad)

    def __enter__(self) -> GradientSiteMaskGuard:
        if self._entered or self._closed:
            raise GradientSiteMaskGuardError("Gradient site-mask guard is one-shot.")
        self._validate_topology(source_basis_required=True)
        try:
            for role in ROLES:
                selected = set(self.mask.for_role(role).selected_site_rows)
                unselected = tuple(
                    row for row in range(EXACT_SITE_COUNT) if row not in selected
                )
                if not unselected:
                    raise GradientSiteMaskGuardError(
                        f"Role {role.upper()} unexpectedly selected every site."
                    )
                self._unselected_rows[role] = unselected
                index = torch.tensor(unselected, dtype=torch.long)
                left = getattr(self.registry, f"{role}_left")
                right = getattr(self.registry, f"{role}_right")
                if not isinstance(left, nn.Parameter) or not isinstance(
                    right, nn.Parameter
                ):
                    raise GradientSiteMaskGuardError(
                        "Guided registry factors must remain Parameters."
                    )
                if int(left.shape[0]) != EXACT_SITE_COUNT or int(
                    right.shape[0]
                ) != EXACT_SITE_COUNT:
                    raise GradientSiteMaskGuardError(
                        "Guided factor row count differs from the full audit."
                    )
                left_rows = left.detach().cpu().index_select(0, index).contiguous()
                if not _exact_zero_bytes(left_rows):
                    raise GradientSiteMaskGuardError(
                        f"Unselected {role.upper()} left rows are not byte-exact zero."
                    )
                self._unselected_left_sha256s[role] = _tensor_sha256(left_rows)
                right_rows = right.detach().cpu().index_select(0, index).contiguous()
                self._unselected_right_sha256s[role] = _tensor_sha256(right_rows)
                for side, parameter in (("left", left), ("right", right)):
                    name = f"{role}_{side}"
                    self._parameter_signatures[name] = self._signature(parameter)
                    self._handles.append(self._register_hook(parameter, unselected))
            site_indices = self.registry.site_indices
            self._site_indices_signature = (
                id(site_indices),
                int(site_indices.data_ptr()),
                int(site_indices._version),
                _tensor_sha256(site_indices),
            )
            self._entered = True
            return self
        except BaseException:
            self._remove_hooks()
            self._closed = True
            raise

    def verify_step(self) -> None:
        if not self._entered or self._closed:
            raise GradientSiteMaskGuardError("Gradient site-mask guard is not active.")
        self._validate_topology(source_basis_required=False)
        site_indices = self.registry.site_indices
        current_site_signature = (
            id(site_indices),
            int(site_indices.data_ptr()),
            int(site_indices._version),
            _tensor_sha256(site_indices),
        )
        if current_site_signature != self._site_indices_signature:
            raise GradientSiteMaskGuardError(
                "Guided registry site-index ownership or bytes changed."
            )
        for role in ROLES:
            rows = torch.tensor(self._unselected_rows[role], dtype=torch.long)
            left = getattr(self.registry, f"{role}_left")
            right = getattr(self.registry, f"{role}_right")
            for side, parameter in (("left", left), ("right", right)):
                if not isinstance(parameter, nn.Parameter) or self._signature(
                    parameter
                ) != self._parameter_signatures[f"{role}_{side}"]:
                    raise GradientSiteMaskGuardError(
                        f"Guided {role.upper()} {side} ownership/storage changed."
                    )
            left_rows = left.detach().cpu().index_select(0, rows).contiguous()
            if (
                not _exact_zero_bytes(left_rows)
                or _tensor_sha256(left_rows)
                != self._unselected_left_sha256s[role]
            ):
                raise GradientSiteMaskGuardError(
                    f"Unselected {role.upper()} left rows changed from exact zero."
                )
            if _tensor_sha256(
                right.detach().cpu().index_select(0, rows).contiguous()
            ) != self._unselected_right_sha256s[role]:
                raise GradientSiteMaskGuardError(
                    f"Unselected {role.upper()} right rows changed from source basis."
                )

    def _remove_hooks(self) -> None:
        handles, self._handles = self._handles, []
        for handle in handles:
            handle.remove()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        del traceback
        verification_error: BaseException | None = None
        try:
            if self._entered and not self._closed:
                self.verify_step()
        except BaseException as error:
            verification_error = error
        finally:
            self._remove_hooks()
            self._entered = False
            self._closed = True
        if exc is not None:
            if verification_error is not None and hasattr(exc, "add_note"):
                exc.add_note(
                    "Gradient-site-mask exit verification also failed: "
                    f"{verification_error!r}"
                )
            return False
        if verification_error is not None:
            raise verification_error
        del exc_type
        return False


__all__ = [
    "EXACT_SITE_COUNT",
    "GUIDED_ATTENTION_ROLE_MASK_SCHEMA",
    "GUIDED_ATTENTION_SITE_MASK_SCHEMA",
    "GradientSiteMaskGuard",
    "GradientSiteMaskGuardError",
    "GuidedAttentionRoleSiteMask",
    "GuidedAttentionSiteMask",
    "GuidedAttentionSiteMaskAuthenticationError",
    "GuidedAttentionSiteMaskError",
    "MINIMUM_SELECTED_SITES",
    "NORMAL_ENERGY_COVERAGE",
    "ROLES",
    "build_guided_attention_site_mask",
]
