"""Target-native low-rank corrections for dense SwiGLU projections.

The temporary operator corrects all three roles of a native SwiGLU MLP::

    gate(x) += U_gate @ (R_gate @ x)
    up(x)   += U_up   @ (R_up   @ x)
    down(h) += U_down @ (R_down @ h)

Every output factor ``U`` starts at exact zero, so installation is an exact
identity transaction.  The factors are target-shaped, have no parent runtime
references, and can be absorbed into the existing dense Gate/Up/Down weights.
Publication therefore restores the original state-key topology and parameter
count instead of appending residual neurons to the child.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


SwiGLUProjectionRole = Literal["gate", "up", "down"]
GuidedSwiGLUTrainingStage = Literal["guided-left", "guided", "joint"]

DENSE_SWIGLU_PROJECTION_RESIDUAL_ACCOUNTING_SCHEMA = (
    "neural-forge-dense-swiglu-projection-residual-accounting-v1"
)
DENSE_SWIGLU_PROJECTION_RESIDUAL_INSTALLATION_SCHEMA = (
    "neural-forge-dense-swiglu-projection-residual-installation-v2"
)
DENSE_SWIGLU_PROJECTION_RESIDUAL_BAKE_SCHEMA = (
    "neural-forge-dense-swiglu-projection-residual-bake-v1"
)
GUIDED_DENSE_SWIGLU_PROJECTION_RESIDUAL_REGISTRY_SCHEMA = (
    "neural-forge-guided-dense-swiglu-projection-residual-registry-v1"
)
GUIDED_DENSE_SWIGLU_PROJECTION_RESIDUAL_INSTALLATION_SCHEMA = (
    "neural-forge-guided-dense-swiglu-projection-residual-installation-v1"
)
GUIDED_DENSE_SWIGLU_PROJECTION_RESIDUAL_BAKE_SCHEMA = (
    "neural-forge-guided-dense-swiglu-projection-residual-bake-v1"
)


class DenseSwiGLUProjectionResidualError(RuntimeError):
    """A target-native SwiGLU projection correction is invalid."""


class DenseSwiGLUProjectionResidualTopologyError(DenseSwiGLUProjectionResidualError):
    """The dense MLP topology or transaction ownership changed."""


class DenseSwiGLUProjectionResidualAccountingError(DenseSwiGLUProjectionResidualError):
    """Closed-form and live parameter accounting disagree."""


class DenseSwiGLUProjectionResidualTrainingError(DenseSwiGLUProjectionResidualError):
    """The residual-only output-training boundary was violated."""


def _index(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DenseSwiGLUProjectionResidualTopologyError(f"{name} must be an integer.")
    if (positive and value <= 0) or (not positive and value < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise DenseSwiGLUProjectionResidualTopologyError(f"{name} must be {qualifier}.")
    return value


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise DenseSwiGLUProjectionResidualError(
            "SwiGLU residual provenance must be canonical JSON."
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(contiguous.dtype), "shape": list(contiguous.shape)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    payload = contiguous.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + b"\0" + payload).hexdigest()


def _required_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DenseSwiGLUProjectionResidualTopologyError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _swiglu_factor_state_sha256(
    registry: SparseDenseSwiGLUProjectionResidualRegistry,
) -> str:
    return _canonical_sha256(
        [
            {
                "name": name,
                "sha256": _tensor_sha256(value),
            }
            for name, value in registry.named_parameters(recurse=False)
        ]
    )


def _unique_parameter_count(module: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in module.parameters():
        if id(parameter) not in seen:
            seen.add(id(parameter))
            total += int(parameter.numel())
    return total


@dataclass(frozen=True, slots=True)
class DenseSwiGLUProjectionResidualAccounting:
    hidden_size: int
    intermediate_size: int
    rank: int
    site_count: int
    gate_adapter_parameters: int
    up_adapter_parameters: int
    down_adapter_parameters: int
    adapter_parameters: int
    installed_parameter_delta: int
    baked_publication_parameter_delta: int

    @classmethod
    def build(
        cls,
        *,
        hidden_size: int,
        intermediate_size: int,
        rank: int,
        site_count: int,
    ) -> DenseSwiGLUProjectionResidualAccounting:
        hidden = _index(hidden_size, "hidden_size", positive=True)
        intermediate = _index(intermediate_size, "intermediate_size", positive=True)
        rank_value = _index(rank, "rank", positive=True)
        sites = _index(site_count, "site_count", positive=True)
        per_role = sites * rank_value * (hidden + intermediate)
        total = 3 * per_role
        return cls(
            hidden_size=hidden,
            intermediate_size=intermediate,
            rank=rank_value,
            site_count=sites,
            gate_adapter_parameters=per_role,
            up_adapter_parameters=per_role,
            down_adapter_parameters=per_role,
            adapter_parameters=total,
            installed_parameter_delta=total,
            baked_publication_parameter_delta=0,
        )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": DENSE_SWIGLU_PROJECTION_RESIDUAL_ACCOUNTING_SCHEMA,
            **{name: getattr(self, name) for name in self.__dataclass_fields__},
            "role_complete_gate_up_down": True,
            "formula": "site_count*rank*3*(hidden_size+intermediate_size)",
            "bake_rule": "dense_weight += left_factor @ right_factor",
            "published_adapter_parameter_count": 0,
            "parent_runtime_reference_count": 0,
        }
        payload["accounting_sha256"] = _canonical_sha256(payload)
        return payload


class SparseDenseSwiGLUProjectionResidualRegistry(nn.Module):
    """Low-rank Gate/Up/Down factors for selected target layers."""

    _ROLES: tuple[SwiGLUProjectionRole, ...] = ("gate", "up", "down")

    def __init__(
        self,
        layer_indices: Sequence[int],
        *,
        hidden_size: int,
        intermediate_size: int,
        rank: int,
        dtype: torch.dtype,
        device: torch.device | str,
        initialization_seed: int = 0,
    ) -> None:
        super().__init__()
        layers = tuple(_index(value, "layer_index") for value in layer_indices)
        if not layers or layers != tuple(sorted(set(layers))):
            raise DenseSwiGLUProjectionResidualTopologyError(
                "layer_indices must be nonempty, sorted, and unique."
            )
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Residual factor dtype must be floating point."
            )
        self.hidden_size = _index(hidden_size, "hidden_size", positive=True)
        self.intermediate_size = _index(
            intermediate_size, "intermediate_size", positive=True
        )
        self.rank = _index(rank, "rank", positive=True)
        self.register_buffer(
            "layer_indices",
            torch.tensor(layers, dtype=torch.int32, device=device),
        )
        self._row_by_layer = {layer: row for row, layer in enumerate(layers)}
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_index(initialization_seed, "initialization_seed"))
        widths = {
            "gate": (self.intermediate_size, self.hidden_size),
            "up": (self.intermediate_size, self.hidden_size),
            "down": (self.hidden_size, self.intermediate_size),
        }
        for role in self._ROLES:
            output_width, input_width = widths[role]
            left = torch.zeros(
                (len(layers), output_width, self.rank),
                dtype=dtype,
                device=device,
            )
            right = torch.empty(
                (len(layers), self.rank, input_width), dtype=torch.float32
            )
            right.uniform_(
                -1.0 / math.sqrt(input_width),
                1.0 / math.sqrt(input_width),
                generator=generator,
            )
            setattr(self, f"{role}_left", nn.Parameter(left))
            setattr(
                self,
                f"{role}_right",
                nn.Parameter(right.to(device=device, dtype=dtype)),
            )
        # Correction is called once per role/layer, so a full factor scan in
        # that hot path would reread the entire registry many times during one
        # model forward.  Parameter versions are incremented by normal
        # in-place updates (including optimizer and state-restore ``copy_``
        # operations).  Cache a scan only after a stable, complete validation;
        # any ownership/storage/version change invalidates it.
        self._finite_parameter_signature_cache: (
            tuple[tuple[int, int, int], ...] | None
        ) = None
        self.assert_finite()

    @property
    def site_count(self) -> int:
        return int(self.layer_indices.numel())

    @property
    def learned_parameter_count(self) -> int:
        return sum(int(value.numel()) for value in self.parameters())

    @property
    def starts_as_exact_zero_correction(self) -> bool:
        return all(
            int(torch.count_nonzero(getattr(self, f"{role}_left")).item()) == 0
            for role in self._ROLES
        )

    @property
    def input_factors_are_nonzero(self) -> bool:
        return all(
            int(torch.count_nonzero(getattr(self, f"{role}_right")).item()) > 0
            for role in self._ROLES
        )

    def output_trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.parameters())

    @staticmethod
    def _finite_parameter_signature(
        parameters: Sequence[nn.Parameter],
    ) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (id(value), int(value.data_ptr()), int(value._version))
            for value in parameters
        )

    def _scan_all_factors_finite(self) -> None:
        """Validate every factor and cache only one stable parameter version."""

        parameters = tuple(self.parameters())
        before = self._finite_parameter_signature(parameters)
        if any(not bool(torch.isfinite(value.detach()).all()) for value in parameters):
            self._finite_parameter_signature_cache = None
            raise DenseSwiGLUProjectionResidualError(
                "SwiGLU projection factors contain NaN or infinity."
            )
        after_parameters = tuple(self.parameters())
        after = self._finite_parameter_signature(after_parameters)
        if (
            tuple(id(value) for value in after_parameters)
            != tuple(id(value) for value in parameters)
            or after != before
        ):
            self._finite_parameter_signature_cache = None
            raise DenseSwiGLUProjectionResidualError(
                "SwiGLU projection factors changed during finite validation."
            )
        self._finite_parameter_signature_cache = after

    def _assert_finite_if_versions_changed(self) -> None:
        parameters = tuple(self.parameters())
        if (
            self._finite_parameter_signature(parameters)
            != self._finite_parameter_signature_cache
        ):
            self._scan_all_factors_finite()

    def assert_finite(self) -> None:
        """Explicitly rescan every factor, independent of the hot-path cache."""

        self._scan_all_factors_finite()

    def _row(self, layer_index: int) -> int:
        layer = _index(layer_index, "layer_index")
        if layer not in self._row_by_layer:
            raise DenseSwiGLUProjectionResidualTopologyError(
                "layer_index is not an active SwiGLU residual site."
            )
        return self._row_by_layer[layer]

    def correction(
        self,
        role: SwiGLUProjectionRole,
        layer_index: int,
        inputs: Tensor,
    ) -> Tensor:
        if role not in self._ROLES:
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Projection role must be gate, up, or down."
            )
        self._assert_finite_if_versions_changed()
        row = self._row(layer_index)
        right = getattr(self, f"{role}_right")[row]
        left = getattr(self, f"{role}_left")[row]
        return F.linear(F.linear(inputs.to(dtype=right.dtype), right), left)

    def dense_delta(self, role: SwiGLUProjectionRole, layer_index: int) -> Tensor:
        if role not in self._ROLES:
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Projection role must be gate, up, or down."
            )
        self.assert_finite()
        row = self._row(layer_index)
        result = (
            getattr(self, f"{role}_left")[row] @ getattr(self, f"{role}_right")[row]
        )
        if not bool(torch.isfinite(result.detach()).all()):
            raise DenseSwiGLUProjectionResidualError(
                "Materialized SwiGLU projection correction is non-finite."
            )
        return result


class GuidedSparseDenseSwiGLUProjectionResidualRegistry(
    SparseDenseSwiGLUProjectionResidualRegistry
):
    """A separate exact-zero G/U/D product with authenticated right rows."""

    def __init__(
        self,
        layer_indices: Sequence[int],
        *,
        hidden_size: int,
        intermediate_size: int,
        rank: int,
        guided_right_rows: Mapping[SwiGLUProjectionRole, Tensor],
        source_basis_tensor_sha256s: Mapping[SwiGLUProjectionRole, str],
        audit_report_sha256: str,
        basis_envelope_file_sha256: str,
        basis_payload_sha256: str,
        basis_manifest_sha256: str,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        if (
            not isinstance(guided_right_rows, Mapping)
            or tuple(guided_right_rows) != self._ROLES
        ):
            raise DenseSwiGLUProjectionResidualTopologyError(
                "guided_right_rows must be ordered gate, up, down rows."
            )
        if (
            not isinstance(source_basis_tensor_sha256s, Mapping)
            or tuple(source_basis_tensor_sha256s) != self._ROLES
        ):
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Source basis hashes must be ordered gate, up, down digests."
            )
        layers = tuple(_index(value, "layer_index") for value in layer_indices)
        if not layers or layers != tuple(sorted(set(layers))):
            raise DenseSwiGLUProjectionResidualTopologyError(
                "layer_indices must be nonempty, sorted, and unique."
            )
        hidden = _index(hidden_size, "hidden_size", positive=True)
        intermediate = _index(intermediate_size, "intermediate_size", positive=True)
        rank_value = _index(rank, "rank", positive=True)
        expected_input_width = {
            "gate": hidden,
            "up": hidden,
            "down": intermediate,
        }
        authenticated_rows: dict[SwiGLUProjectionRole, Tensor] = {}
        authenticated_hashes: dict[SwiGLUProjectionRole, str] = {}
        for role in self._ROLES:
            value = guided_right_rows[role]
            expected_shape = (len(layers), rank_value, expected_input_width[role])
            if (
                not isinstance(value, Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.float32
                or not value.is_contiguous()
                or tuple(value.shape) != expected_shape
                or not bool(torch.isfinite(value).all())
            ):
                raise DenseSwiGLUProjectionResidualTopologyError(
                    f"Guided {role} right rows must be finite contiguous CPU FP32 "
                    f"with shape {expected_shape}."
                )
            digest = _tensor_sha256(value)
            expected_digest = _required_sha256(
                source_basis_tensor_sha256s[role],
                f"source {role} basis tensor SHA-256",
            )
            if digest != expected_digest:
                raise DenseSwiGLUProjectionResidualTopologyError(
                    f"Guided {role} right rows failed source authentication."
                )
            authenticated_rows[role] = value
            authenticated_hashes[role] = digest
        if dtype != torch.float32:
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Guided SwiGLU factors must be installed in FP32."
            )
        super().__init__(
            layers,
            hidden_size=hidden,
            intermediate_size=intermediate,
            rank=rank_value,
            dtype=dtype,
            device=device,
            initialization_seed=0,
        )
        with torch.no_grad():
            for role in self._ROLES:
                getattr(self, f"{role}_right").copy_(
                    authenticated_rows[role].to(device=device, dtype=dtype)
                )
        self.audit_report_sha256 = _required_sha256(
            audit_report_sha256, "audit_report_sha256"
        )
        self.basis_envelope_file_sha256 = _required_sha256(
            basis_envelope_file_sha256, "basis_envelope_file_sha256"
        )
        self.basis_payload_sha256 = _required_sha256(
            basis_payload_sha256, "basis_payload_sha256"
        )
        self.basis_manifest_sha256 = _required_sha256(
            basis_manifest_sha256, "basis_manifest_sha256"
        )
        self.source_basis_tensor_sha256s = dict(authenticated_hashes)
        self.assert_finite()
        if not self.starts_as_exact_zero_correction:
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Guided SwiGLU left factors must start at exact zero."
            )
        if any(
            _tensor_sha256(getattr(self, f"{role}_right")) != authenticated_hashes[role]
            for role in self._ROLES
        ):
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Device installation changed an authenticated guided right basis."
            )

    def as_dict(self) -> dict[str, Any]:
        role_parameters = {
            role: int(
                getattr(self, f"{role}_left").numel()
                + getattr(self, f"{role}_right").numel()
            )
            for role in self._ROLES
        }
        payload = {
            "schema": GUIDED_DENSE_SWIGLU_PROJECTION_RESIDUAL_REGISTRY_SCHEMA,
            "layer_indices": self.layer_indices.detach().cpu().tolist(),
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "rank": self.rank,
            "site_count": self.site_count,
            "learned_parameter_count": self.learned_parameter_count,
            "role_parameter_counts": role_parameters,
            "zero_guided_left_initialization": self.starts_as_exact_zero_correction,
            "authenticated_guided_right_initialization": True,
            "audit_report_sha256": self.audit_report_sha256,
            "basis_envelope_file_sha256": self.basis_envelope_file_sha256,
            "basis_payload_sha256": self.basis_payload_sha256,
            "basis_manifest_sha256": self.basis_manifest_sha256,
            "source_basis_tensor_sha256s": dict(self.source_basis_tensor_sha256s),
            "separate_from_inherited_registry": True,
            "old_and_guided_factors_concatenated": False,
            "old_and_guided_factors_regrouped": False,
            "target_shaped_factors_only": True,
            "parent_runtime_reference_count": 0,
            "cross_token_mixing_inside_adapter": False,
        }
        payload["registry_sha256"] = _canonical_sha256(payload)
        return payload


class DenseSwiGLUResidualProjection(nn.Module):
    """Linear-compatible projection plus one target-native correction."""

    def __init__(
        self,
        base_projection: nn.Linear,
        registry: SparseDenseSwiGLUProjectionResidualRegistry,
        *,
        layer_index: int,
        role: SwiGLUProjectionRole,
    ) -> None:
        super().__init__()
        if type(base_projection) is not nn.Linear or role not in registry._ROLES:
            raise DenseSwiGLUProjectionResidualTopologyError(
                "A plain dense projection and valid SwiGLU role are required."
            )
        object.__setattr__(self, "_registry", registry)
        object.__setattr__(self, "_guided_registry", None)
        self.base_projection = base_projection
        self.layer_index = _index(layer_index, "layer_index")
        self.role = role
        self.in_features = base_projection.in_features
        self.out_features = base_projection.out_features
        self.train(base_projection.training)

    @property
    def registry(self) -> SparseDenseSwiGLUProjectionResidualRegistry:
        return self._registry

    @property
    def guided_registry(
        self,
    ) -> GuidedSparseDenseSwiGLUProjectionResidualRegistry | None:
        return self._guided_registry

    @property
    def guided_adapter_registry(
        self,
    ) -> GuidedSparseDenseSwiGLUProjectionResidualRegistry | None:
        """Compatibility spelling shared with the guided QKVO wrapper."""

        return self._guided_registry

    def _attach_guided_registry(
        self,
        registry: GuidedSparseDenseSwiGLUProjectionResidualRegistry,
    ) -> None:
        if not isinstance(registry, GuidedSparseDenseSwiGLUProjectionResidualRegistry):
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Only an authenticated guided SwiGLU registry can be attached."
            )
        if self._guided_registry is not None:
            raise DenseSwiGLUProjectionResidualTopologyError(
                "A guided SwiGLU tranche is already attached."
            )
        if (
            registry.hidden_size != self.registry.hidden_size
            or registry.intermediate_size != self.registry.intermediate_size
            or registry.rank != self.registry.rank
            or not torch.equal(
                registry.layer_indices.detach().cpu(),
                self.registry.layer_indices.detach().cpu(),
            )
        ):
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Guided and inherited SwiGLU registry topology differs."
            )
        object.__setattr__(self, "_guided_registry", registry)

    def _detach_guided_registry(
        self,
        registry: GuidedSparseDenseSwiGLUProjectionResidualRegistry,
    ) -> None:
        if self._guided_registry is not registry:
            raise DenseSwiGLUProjectionResidualTopologyError(
                "The attached guided SwiGLU registry changed."
            )
        object.__setattr__(self, "_guided_registry", None)

    @property
    def weight(self) -> Tensor:
        return self.base_projection.weight

    @property
    def bias(self) -> Tensor | None:
        return self.base_projection.bias

    def forward(self, inputs: Tensor) -> Tensor:
        base = self.base_projection(inputs)
        correction = self.registry.correction(self.role, self.layer_index, inputs).to(
            dtype=base.dtype
        )
        if tuple(base.shape) != tuple(correction.shape):
            raise DenseSwiGLUProjectionResidualTopologyError(
                "SwiGLU correction output shape does not match its base projection."
            )
        result = base + correction
        if self.guided_registry is not None:
            guided = self.guided_registry.correction(
                self.role, self.layer_index, inputs
            ).to(dtype=base.dtype)
            if tuple(result.shape) != tuple(guided.shape):
                raise DenseSwiGLUProjectionResidualTopologyError(
                    "Guided SwiGLU correction output shape does not match."
                )
            result = result + guided
        return result


@dataclass(frozen=True, slots=True)
class DenseSwiGLUProjectionResidualInstallationReport:
    layer_indices: tuple[int, ...]
    hidden_size: int
    intermediate_size: int
    rank: int
    factor_dtype: str
    owner_name: str
    projection_names: Mapping[str, str]
    adapter_parameter_count: int
    model_parameter_count_before: int
    model_parameter_count_installed: int
    base_projection_state_sha256: str
    accounting: DenseSwiGLUProjectionResidualAccounting

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": DENSE_SWIGLU_PROJECTION_RESIDUAL_INSTALLATION_SCHEMA,
            "layer_indices": list(self.layer_indices),
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "rank": self.rank,
            "factor_dtype": self.factor_dtype,
            "owner_name": self.owner_name,
            "projection_names": dict(self.projection_names),
            "adapter_parameter_count": self.adapter_parameter_count,
            "model_parameter_count_before": self.model_parameter_count_before,
            "model_parameter_count_installed": self.model_parameter_count_installed,
            "base_projection_state_sha256": self.base_projection_state_sha256,
            "accounting": self.accounting.as_dict(),
            "role_complete_gate_up_down": True,
            "zero_step_identity": True,
            "internal_numeric_matching_used": False,
            "output_function_training_only": True,
            "native_dense_bake_supported": True,
            "checkpoint_publication_allowed_while_installed": False,
        }
        payload["installation_sha256"] = _canonical_sha256(payload)
        return payload


@dataclass(frozen=True, slots=True)
class DenseSwiGLUProjectionResidualBakeReport:
    layer_indices: tuple[int, ...]
    rank: int
    adapter_parameters_removed: int
    model_parameter_count_before_installation: int
    model_parameter_count_before_bake: int
    model_parameter_count_after_bake: int
    dense_delta_sha256: str
    affected_projection_count: int

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": DENSE_SWIGLU_PROJECTION_RESIDUAL_BAKE_SCHEMA,
            "layer_indices": list(self.layer_indices),
            "rank": self.rank,
            "adapter_parameters_removed": self.adapter_parameters_removed,
            "model_parameter_count_before_installation": (
                self.model_parameter_count_before_installation
            ),
            "model_parameter_count_before_bake": self.model_parameter_count_before_bake,
            "model_parameter_count_after_bake": self.model_parameter_count_after_bake,
            "dense_delta_sha256": self.dense_delta_sha256,
            "affected_projection_count": self.affected_projection_count,
            "role_complete_gate_up_down": True,
            "published_projection_type": "torch.nn.Linear",
            "published_adapter_parameter_count": 0,
            "published_parameter_delta_vs_original_dense_target": 0,
            "internal_numeric_matching_used": False,
        }
        payload["bake_sha256"] = _canonical_sha256(payload)
        return payload


@dataclass(frozen=True, slots=True)
class GuidedDenseSwiGLUProjectionResidualInstallationReport:
    """Authenticated evidence for separate inherited and guided products."""

    layer_indices: tuple[int, ...]
    hidden_size: int
    intermediate_size: int
    rank: int
    factor_dtype: str
    owner_name: str
    projection_names: Mapping[str, str]
    inherited_adapter_parameter_count: int
    guided_adapter_parameter_count: int
    adapter_parameter_count: int
    model_parameter_count_before: int
    model_parameter_count_before_guided_installation: int
    model_parameter_count_installed: int
    adapter_layer_indices_sha256: str
    base_projection_state_sha256: str
    audit_report_sha256: str
    basis_envelope_file_sha256: str
    basis_payload_sha256: str
    basis_manifest_sha256: str
    source_basis_tensor_sha256s: Mapping[str, str]
    inherited_installation_sha256: str
    inherited_factor_state_sha256_at_install: str
    inherited_handle_identity: int
    inherited_registry_identity: int

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": (GUIDED_DENSE_SWIGLU_PROJECTION_RESIDUAL_INSTALLATION_SCHEMA),
            "layer_indices": list(self.layer_indices),
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "rank": self.rank,
            "factor_dtype": self.factor_dtype,
            "owner_name": self.owner_name,
            "projection_names": dict(self.projection_names),
            "inherited_adapter_parameter_count": (
                self.inherited_adapter_parameter_count
            ),
            "guided_adapter_parameter_count": self.guided_adapter_parameter_count,
            "adapter_parameter_count": self.adapter_parameter_count,
            "model_parameter_count_before": self.model_parameter_count_before,
            "model_parameter_count_before_guided_installation": (
                self.model_parameter_count_before_guided_installation
            ),
            "model_parameter_count_installed": self.model_parameter_count_installed,
            "adapter_layer_indices_sha256": self.adapter_layer_indices_sha256,
            "base_projection_state_sha256": self.base_projection_state_sha256,
            "audit_report_sha256": self.audit_report_sha256,
            "basis_envelope_file_sha256": self.basis_envelope_file_sha256,
            "basis_payload_sha256": self.basis_payload_sha256,
            "basis_manifest_sha256": self.basis_manifest_sha256,
            "source_basis_tensor_sha256s": dict(self.source_basis_tensor_sha256s),
            "inherited_installation_sha256": self.inherited_installation_sha256,
            "inherited_factor_state_sha256_at_install": (
                self.inherited_factor_state_sha256_at_install
            ),
            "inherited_handle_identity": self.inherited_handle_identity,
            "inherited_registry_identity": self.inherited_registry_identity,
            "initial_training_stage": "guided-left",
            "forward_product_order": ["inherited", "guided"],
            "bake_product_order": ["inherited", "guided"],
            "factor_concatenation_used": False,
            "factor_regrouping_used": False,
            "zero_guided_left_preserves_inherited_function_exactly": True,
            "role_complete_gate_up_down": True,
            "pointwise_causal_operator": True,
            "cache_sidecar_required": False,
            "parent_runtime_reference_count": 0,
            "bake_to_plain_dense_weights_supported": True,
            "checkpoint_saved": False,
            "candidate_committed": False,
        }
        payload["installation_sha256"] = _canonical_sha256(payload)
        return payload


@dataclass(frozen=True, slots=True)
class GuidedDenseSwiGLUProjectionResidualBakeReport:
    """Publication evidence for inherited-then-guided native G/U/D bake."""

    layer_indices: tuple[int, ...]
    rank: int
    inherited_adapter_parameters_removed: int
    guided_adapter_parameters_removed: int
    adapter_parameters_removed: int
    model_parameter_count_before_installation: int
    model_parameter_count_before_bake: int
    model_parameter_count_after_bake: int
    inherited_dense_delta_sha256: str
    guided_dense_delta_sha256: str
    sequential_dense_delta_sha256: str
    affected_projection_count: int

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": GUIDED_DENSE_SWIGLU_PROJECTION_RESIDUAL_BAKE_SCHEMA,
            "layer_indices": list(self.layer_indices),
            "rank": self.rank,
            "inherited_adapter_parameters_removed": (
                self.inherited_adapter_parameters_removed
            ),
            "guided_adapter_parameters_removed": (
                self.guided_adapter_parameters_removed
            ),
            "adapter_parameters_removed": self.adapter_parameters_removed,
            "model_parameter_count_before_installation": (
                self.model_parameter_count_before_installation
            ),
            "model_parameter_count_before_bake": self.model_parameter_count_before_bake,
            "model_parameter_count_after_bake": self.model_parameter_count_after_bake,
            "inherited_dense_delta_sha256": self.inherited_dense_delta_sha256,
            "guided_dense_delta_sha256": self.guided_dense_delta_sha256,
            "sequential_dense_delta_sha256": self.sequential_dense_delta_sha256,
            "dense_delta_sha256": self.sequential_dense_delta_sha256,
            "affected_projection_count": self.affected_projection_count,
            "bake_rule": [
                "dense_weight += inherited_left @ inherited_right",
                "dense_weight += guided_left @ guided_right",
            ],
            "factor_concatenation_used": False,
            "factor_regrouping_used": False,
            "role_complete_gate_up_down": True,
            "published_projection_type": "torch.nn.Linear",
            "published_adapter_parameter_count": 0,
            "published_parameter_delta_vs_original_dense_target": 0,
            "published_adapter_flops_per_token": 0,
            "inherited_registry_removed": True,
            "guided_registry_removed": True,
            "projection_wrappers_removed": True,
            "parent_runtime_reference_count": 0,
            "source_sized_sidecar_retained": False,
            "functional_publication_gate_still_required": True,
        }
        payload["bake_sha256"] = _canonical_sha256(payload)
        return payload


@dataclass(frozen=True, slots=True)
class _TensorSnapshot:
    tensor: Tensor
    data_pointer: int
    version: int
    sha256: str


def _projection_state(
    projections: Mapping[tuple[int, SwiGLUProjectionRole], nn.Linear],
) -> dict[tuple[int, SwiGLUProjectionRole, str], _TensorSnapshot]:
    result: dict[tuple[int, SwiGLUProjectionRole, str], _TensorSnapshot] = {}
    for (layer, role), projection in projections.items():
        for name, tensor in (("weight", projection.weight), ("bias", projection.bias)):
            if tensor is None:
                continue
            result[(layer, role, name)] = _TensorSnapshot(
                tensor=tensor,
                data_pointer=int(tensor.data_ptr()),
                version=int(tensor._version),
                sha256=_tensor_sha256(tensor),
            )
    return result


def _projection_state_sha256(
    state: Mapping[tuple[int, SwiGLUProjectionRole, str], _TensorSnapshot],
) -> str:
    return _canonical_sha256(
        [
            {
                "layer": layer,
                "role": role,
                "tensor": name,
                "sha256": snapshot.sha256,
            }
            for (layer, role, name), snapshot in sorted(state.items())
        ]
    )


def _normalized_projection_names(
    value: Mapping[str, str] | None,
) -> dict[str, str]:
    result = (
        {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}
        if value is None
        else dict(value)
    )
    if set(result) != {"gate", "up", "down"} or any(
        not isinstance(name, str) or not name.isidentifier() for name in result.values()
    ):
        raise DenseSwiGLUProjectionResidualTopologyError(
            "projection_names must map gate, up, and down to identifiers."
        )
    return result


class InstalledDenseSwiGLUProjectionResidualAdapter:
    """Live role-complete MLP adapter with rollback and native bake."""

    def __init__(
        self,
        *,
        model: nn.Module,
        mlp_modules: Mapping[int, nn.Module],
        registry: SparseDenseSwiGLUProjectionResidualRegistry,
        report: DenseSwiGLUProjectionResidualInstallationReport,
        originals: Mapping[tuple[int, SwiGLUProjectionRole], nn.Linear],
        installed: Mapping[
            tuple[int, SwiGLUProjectionRole], DenseSwiGLUResidualProjection
        ],
        projection_state_before: Mapping[
            tuple[int, SwiGLUProjectionRole, str], _TensorSnapshot
        ],
    ) -> None:
        self.model = model
        self.mlp_modules = dict(mlp_modules)
        self.registry = registry
        self.report = report
        self._originals = dict(originals)
        self._installed = dict(installed)
        self._projection_state_before = dict(projection_state_before)
        self._published: dict[tuple[int, SwiGLUProjectionRole], nn.Linear] = {}
        self._state: Literal["live", "rolled-back", "baked"] = "live"
        self._bake_report: DenseSwiGLUProjectionResidualBakeReport | None = None

    @property
    def is_applied(self) -> bool:
        return self._state == "live"

    @property
    def is_baked(self) -> bool:
        return self._state == "baked"

    def output_trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        self.assert_live()
        return self.registry.output_trainable_parameters()

    def _assert_base_unchanged(self) -> None:
        current = _projection_state(self._originals)
        if set(current) != set(self._projection_state_before):
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Dense SwiGLU base topology changed."
            )
        for key, before in self._projection_state_before.items():
            after = current[key]
            if (
                after.tensor is not before.tensor
                or after.data_pointer != before.data_pointer
                or after.version != before.version
                or after.sha256 != before.sha256
            ):
                raise DenseSwiGLUProjectionResidualTopologyError(
                    "A dense SwiGLU base parameter changed during the transaction."
                )

    def assert_structurally_live(
        self,
    ) -> DenseSwiGLUProjectionResidualInstallationReport:
        """Validate ownership/base topology without reading disposable values."""

        if self._state != "live":
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Dense SwiGLU residual adapter is no longer live."
            )
        if getattr(self.model, self.report.owner_name, None) is not self.registry:
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Model lost the SwiGLU residual registry owner."
            )
        for (layer, role), view in self._installed.items():
            if (
                getattr(
                    self.mlp_modules[layer], self.report.projection_names[role], None
                )
                is not view
            ):
                raise DenseSwiGLUProjectionResidualTopologyError(
                    "A live SwiGLU projection wrapper was replaced."
                )
        self._assert_base_unchanged()
        return self.report

    def assert_live(self) -> DenseSwiGLUProjectionResidualInstallationReport:
        self.assert_structurally_live()
        self.registry.assert_finite()
        return self.report

    def output_training_scope(
        self,
    ) -> DenseSwiGLUAdapterOnlyOutputTrainingScope:
        return DenseSwiGLUAdapterOnlyOutputTrainingScope(self)

    def rollback(self) -> None:
        if self._state == "rolled-back":
            return
        if self._state == "baked":
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Use restore_prepublication_native for a partially published bake."
            )
        # Factors are disposable transaction state.  Rollback must remain able
        # to remove them after a failed optimizer update, including when they
        # contain NaN/Inf.  Validate ownership and the immutable dense base, but
        # intentionally do not finite-assert the factors themselves.
        if getattr(self.model, self.report.owner_name, None) is not self.registry:
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Model lost the SwiGLU residual registry owner."
            )
        for (layer, role), view in self._installed.items():
            if (
                getattr(
                    self.mlp_modules[layer], self.report.projection_names[role], None
                )
                is not view
            ):
                raise DenseSwiGLUProjectionResidualTopologyError(
                    "A live SwiGLU projection wrapper was replaced."
                )
        self._assert_base_unchanged()
        for (layer, role), original in self._originals.items():
            setattr(
                self.mlp_modules[layer], self.report.projection_names[role], original
            )
        delattr(self.model, self.report.owner_name)
        self._assert_base_unchanged()
        if (
            _unique_parameter_count(self.model)
            != self.report.model_parameter_count_before
        ):
            raise DenseSwiGLUProjectionResidualTopologyError(
                "SwiGLU rollback parameter accounting failed."
            )
        for (layer, role), original in self._originals.items():
            if (
                getattr(
                    self.mlp_modules[layer], self.report.projection_names[role], None
                )
                is not original
            ):
                raise DenseSwiGLUProjectionResidualTopologyError(
                    "SwiGLU projection rollback failed."
                )
        self._state = "rolled-back"

    def bake_for_publication(self) -> DenseSwiGLUProjectionResidualBakeReport:
        self.assert_live()
        before_bake = _unique_parameter_count(self.model)
        updates: dict[tuple[int, SwiGLUProjectionRole], Tensor] = {}
        for key, original in self._originals.items():
            layer, role = key
            delta = (
                self.registry.dense_delta(role, layer)
                .detach()
                .to(device=original.weight.device, dtype=original.weight.dtype)
            )
            if tuple(delta.shape) != tuple(original.weight.shape):
                raise DenseSwiGLUProjectionResidualTopologyError(
                    "Materialized SwiGLU delta has the wrong dense shape."
                )
            updates[key] = delta
        manifest = [
            {
                "layer": layer,
                "role": role,
                "shape": list(delta.shape),
                "sha256": _tensor_sha256(delta),
            }
            for (layer, role), delta in sorted(updates.items())
        ]
        published: dict[tuple[int, SwiGLUProjectionRole], nn.Linear] = {}
        try:
            with torch.no_grad():
                for key, original in self._originals.items():
                    replacement = nn.Linear(
                        original.in_features,
                        original.out_features,
                        bias=original.bias is not None,
                        device=original.weight.device,
                        dtype=original.weight.dtype,
                    )
                    replacement.weight.copy_(original.weight + updates[key])
                    replacement.weight.requires_grad_(original.weight.requires_grad)
                    if original.bias is not None:
                        assert replacement.bias is not None
                        replacement.bias.copy_(original.bias)
                        replacement.bias.requires_grad_(original.bias.requires_grad)
                    replacement.train(original.training)
                    published[key] = replacement
            if any(
                not bool(torch.isfinite(value.weight.detach()).all())
                for value in published.values()
            ):
                raise DenseSwiGLUProjectionResidualError(
                    "Publication bake produced a non-finite SwiGLU projection."
                )
            for (layer, role), replacement in published.items():
                setattr(
                    self.mlp_modules[layer],
                    self.report.projection_names[role],
                    replacement,
                )
            delattr(self.model, self.report.owner_name)
            after = _unique_parameter_count(self.model)
            if after != self.report.model_parameter_count_before:
                raise DenseSwiGLUProjectionResidualAccountingError(
                    "Baked SwiGLU target did not return to its native parameter count."
                )
        except Exception:
            if not hasattr(self.model, self.report.owner_name):
                self.model.add_module(self.report.owner_name, self.registry)
            for (layer, role), view in self._installed.items():
                setattr(
                    self.mlp_modules[layer], self.report.projection_names[role], view
                )
            raise
        report = DenseSwiGLUProjectionResidualBakeReport(
            layer_indices=self.report.layer_indices,
            rank=self.report.rank,
            adapter_parameters_removed=self.report.adapter_parameter_count,
            model_parameter_count_before_installation=(
                self.report.model_parameter_count_before
            ),
            model_parameter_count_before_bake=before_bake,
            model_parameter_count_after_bake=after,
            dense_delta_sha256=_canonical_sha256(manifest),
            affected_projection_count=len(updates),
        )
        self._published = published
        self._state = "baked"
        self._bake_report = report
        return report

    def restore_prepublication_native(self) -> None:
        """Undo a bake only while a larger atomic transaction is failing."""

        if self._state == "live":
            self.rollback()
            return
        if self._state == "rolled-back":
            return
        for (layer, role), published in self._published.items():
            if (
                getattr(
                    self.mlp_modules[layer], self.report.projection_names[role], None
                )
                is not published
            ):
                raise DenseSwiGLUProjectionResidualTopologyError(
                    "Published SwiGLU ownership changed; restoration is unsafe."
                )
        for (layer, role), original in self._originals.items():
            setattr(
                self.mlp_modules[layer], self.report.projection_names[role], original
            )
        self._assert_base_unchanged()
        if (
            _unique_parameter_count(self.model)
            != self.report.model_parameter_count_before
        ):
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Prepublication SwiGLU restoration changed the native budget."
            )
        self._state = "rolled-back"

    def __enter__(self) -> InstalledDenseSwiGLUProjectionResidualAdapter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        if self._state == "live":
            self.rollback()


class InstalledGuidedDenseSwiGLUProjectionResidualTranche:
    """Transaction owner for separate inherited and guided G/U/D products."""

    _STAGES: tuple[GuidedSwiGLUTrainingStage, ...] = (
        "guided-left",
        "guided",
        "joint",
    )

    def __init__(
        self,
        *,
        inherited_handle: InstalledDenseSwiGLUProjectionResidualAdapter,
        guided_registry: GuidedSparseDenseSwiGLUProjectionResidualRegistry,
        report: GuidedDenseSwiGLUProjectionResidualInstallationReport,
        inherited_requires_grad: Mapping[str, bool],
    ) -> None:
        self.inherited_handle = inherited_handle
        self.guided_registry = guided_registry
        self.report = report
        self.model = inherited_handle.model
        self.mlp_modules = inherited_handle.mlp_modules
        self._inherited_requires_grad = dict(inherited_requires_grad)
        self._state: Literal["live", "rolled-back-guided", "rolled-back", "baked"] = (
            "live"
        )
        self._training_stage: GuidedSwiGLUTrainingStage = "guided-left"
        self._guided_right_training_started = False
        self._bake_report: GuidedDenseSwiGLUProjectionResidualBakeReport | None = None
        self._published: dict[tuple[int, SwiGLUProjectionRole], nn.Linear] = {}
        self._source_binding = MappingProxyType(
            {
                "audit_report_sha256": report.audit_report_sha256,
                "basis_envelope_file_sha256": (report.basis_envelope_file_sha256),
                "basis_payload_sha256": report.basis_payload_sha256,
                "basis_manifest_sha256": report.basis_manifest_sha256,
                "source_basis_tensor_sha256s": MappingProxyType(
                    dict(report.source_basis_tensor_sha256s)
                ),
                "inherited_installation_sha256": (report.inherited_installation_sha256),
                "inherited_factor_state_sha256_at_install": (
                    report.inherited_factor_state_sha256_at_install
                ),
                "inherited_handle_identity": report.inherited_handle_identity,
                "inherited_registry_identity": report.inherited_registry_identity,
                "adapter_layer_indices_sha256": (report.adapter_layer_indices_sha256),
            }
        )
        self._set_requires_grad_for_stage("guided-left")

    @property
    def registry(self) -> SparseDenseSwiGLUProjectionResidualRegistry:
        """Compatibility alias for the inherited adapter registry."""

        return self.inherited_handle.registry

    @property
    def adapter_registry(self) -> SparseDenseSwiGLUProjectionResidualRegistry:
        return self.registry

    @property
    def is_applied(self) -> bool:
        return self._state == "live"

    @property
    def is_baked(self) -> bool:
        return self._state == "baked"

    @property
    def bake_report(self) -> GuidedDenseSwiGLUProjectionResidualBakeReport | None:
        return self._bake_report

    @property
    def training_stage(self) -> GuidedSwiGLUTrainingStage:
        return self._training_stage

    def factor_source_binding(self) -> Mapping[str, Any]:
        return self._source_binding

    def _set_requires_grad_for_stage(self, stage: GuidedSwiGLUTrainingStage) -> None:
        inherited_trainable = stage == "joint"
        for _name, value in self.registry.named_parameters(recurse=False):
            value.requires_grad_(inherited_trainable)
        guided_right_trainable = stage in {"guided", "joint"}
        for name, value in self.guided_registry.named_parameters(recurse=False):
            value.requires_grad_(name.endswith("_left") or guided_right_trainable)

    def set_training_stage(
        self, stage: GuidedSwiGLUTrainingStage
    ) -> tuple[nn.Parameter, ...]:
        self.assert_live()
        if stage not in self._STAGES:
            raise DenseSwiGLUProjectionResidualTrainingError(
                "Guided SwiGLU stage must be guided-left, guided, or joint."
            )
        current_index = self._STAGES.index(self._training_stage)
        requested_index = self._STAGES.index(stage)
        if requested_index < current_index:
            raise DenseSwiGLUProjectionResidualTrainingError(
                "Guided SwiGLU training stages cannot move backward."
            )
        self._training_stage = stage
        if stage in {"guided", "joint"}:
            self._guided_right_training_started = True
        self._set_requires_grad_for_stage(stage)
        return self.output_trainable_parameters()

    def output_trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        self.assert_live()
        inherited = tuple(self.registry.parameters(recurse=False))
        guided = tuple(self.guided_registry.parameters(recurse=False))
        if self._training_stage == "guided-left":
            result = tuple(
                value
                for name, value in self.guided_registry.named_parameters(recurse=False)
                if name.endswith("_left")
            )
        elif self._training_stage == "guided":
            result = guided
        else:
            result = (*inherited, *guided)
        if not result or any(not value.requires_grad for value in result):
            raise DenseSwiGLUProjectionResidualTrainingError(
                "The guided SwiGLU trainable stage is internally inconsistent."
            )
        return result

    def _assert_unique_guided_owner(self) -> None:
        paths = [
            name
            for name, module in self.model.named_modules(remove_duplicate=False)
            if module is self.guided_registry
        ]
        if paths != [self.report.owner_name]:
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Guided SwiGLU registry must have exactly one registered owner."
            )

    def assert_structurally_live(
        self,
    ) -> GuidedDenseSwiGLUProjectionResidualInstallationReport:
        if self._state != "live":
            raise DenseSwiGLUProjectionResidualTopologyError(
                "The guided SwiGLU tranche is no longer live."
            )
        if (
            id(self.inherited_handle) != self.report.inherited_handle_identity
            or id(self.registry) != self.report.inherited_registry_identity
        ):
            raise DenseSwiGLUProjectionResidualTopologyError(
                "The inherited SwiGLU handle identity changed."
            )
        self.inherited_handle.assert_structurally_live()
        if getattr(self.model, self.report.owner_name, None) is not (
            self.guided_registry
        ):
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Model lost the guided SwiGLU registry owner."
            )
        self._assert_unique_guided_owner()
        for installed in self.inherited_handle._installed.values():
            if installed.guided_registry is not self.guided_registry:
                raise DenseSwiGLUProjectionResidualTopologyError(
                    "A projection lost the separate guided SwiGLU product."
                )
        if _tensor_sha256(self.guided_registry.layer_indices) != (
            self.report.adapter_layer_indices_sha256
        ):
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Guided SwiGLU layer indices changed."
            )
        if (
            self._training_stage != "joint"
            and _swiglu_factor_state_sha256(self.registry)
            != self.report.inherited_factor_state_sha256_at_install
        ):
            raise DenseSwiGLUProjectionResidualTopologyError(
                "The inherited SwiGLU product changed before joint training."
            )
        if not self._guided_right_training_started and any(
            _tensor_sha256(getattr(self.guided_registry, f"{role}_right"))
            != self.report.source_basis_tensor_sha256s[role]
            for role in SparseDenseSwiGLUProjectionResidualRegistry._ROLES
        ):
            raise DenseSwiGLUProjectionResidualTopologyError(
                "A guided right basis changed during guided-left-only training."
            )
        return self.report

    def assert_live(
        self,
    ) -> GuidedDenseSwiGLUProjectionResidualInstallationReport:
        self.assert_structurally_live()
        self.registry.assert_finite()
        self.guided_registry.assert_finite()
        return self.report

    def output_training_scope(
        self,
        stage: GuidedSwiGLUTrainingStage | None = None,
    ) -> GuidedDenseSwiGLUProjectionOutputTrainingScope:
        if stage is not None:
            self.set_training_stage(stage)
        return GuidedDenseSwiGLUProjectionOutputTrainingScope(self)

    @contextmanager
    def suspend_guided_registry_owner_for_nested_publication(
        self,
    ) -> Iterator[None]:
        """Temporarily unregister only the guided owner for a nested bake."""

        self.assert_live()
        owner_name = self.report.owner_name
        if getattr(self.model, owner_name, None) is not self.guided_registry:
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Guided registry owner changed before nested publication."
            )
        guided_state_sha256 = _swiglu_factor_state_sha256(self.guided_registry)
        guided_parameter_ownership = tuple(
            (
                name,
                id(value),
                int(value.data_ptr()),
                bool(value.requires_grad),
            )
            for name, value in self.guided_registry.named_parameters(recurse=False)
        )
        before = _unique_parameter_count(self.model)
        delattr(self.model, owner_name)
        suspended = _unique_parameter_count(self.model)
        if suspended != before - self.report.guided_adapter_parameter_count:
            self.model.add_module(owner_name, self.guided_registry)
            raise DenseSwiGLUProjectionResidualAccountingError(
                "Suspending the guided registry changed the wrong count."
            )
        if any(
            module is self.guided_registry
            for _name, module in self.model.named_modules(remove_duplicate=False)
        ):
            self.model.add_module(owner_name, self.guided_registry)
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Guided registry retained an undeclared registered owner."
            )
        try:
            yield
        finally:
            if hasattr(self.model, owner_name):
                raise DenseSwiGLUProjectionResidualTopologyError(
                    "Guided registry owner path was occupied while suspended."
                )
            before_restore = _unique_parameter_count(self.model)
            self.model.add_module(owner_name, self.guided_registry)
            if (
                _unique_parameter_count(self.model)
                != before_restore + self.report.guided_adapter_parameter_count
                or tuple(
                    (
                        name,
                        id(value),
                        int(value.data_ptr()),
                        bool(value.requires_grad),
                    )
                    for name, value in self.guided_registry.named_parameters(
                        recurse=False
                    )
                )
                != guided_parameter_ownership
                or _swiglu_factor_state_sha256(self.guided_registry)
                != guided_state_sha256
            ):
                raise DenseSwiGLUProjectionResidualTopologyError(
                    "Nested publication did not restore the exact guided registry."
                )
            self.assert_live()

    def _rollback_guided_only(
        self,
        *,
        validate_inherited_factors: bool,
    ) -> InstalledDenseSwiGLUProjectionResidualAdapter:
        if self._state == "rolled-back-guided":
            return self.inherited_handle
        if self._state != "live":
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Only a live guided tranche can be removed independently."
            )
        # Guided factors are disposable candidate state.  Rejection must stay
        # possible after a failed optimizer update or source-lock violation,
        # including NaN/Inf or a changed right row.  Authenticate only the
        # inherited handle, wrapper topology, and unique owner here; the live
        # assertion deliberately performs the stronger factor checks.
        if (
            id(self.inherited_handle) != self.report.inherited_handle_identity
            or id(self.registry) != self.report.inherited_registry_identity
        ):
            raise DenseSwiGLUProjectionResidualTopologyError(
                "The inherited SwiGLU handle identity changed."
            )
        self.inherited_handle.assert_structurally_live()
        if getattr(self.model, self.report.owner_name, None) is not (
            self.guided_registry
        ):
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Model lost the guided SwiGLU registry owner."
            )
        self._assert_unique_guided_owner()
        if any(
            installed.guided_registry is not self.guided_registry
            for installed in self.inherited_handle._installed.values()
        ):
            raise DenseSwiGLUProjectionResidualTopologyError(
                "A projection lost the separate guided SwiGLU product."
            )
        if _tensor_sha256(self.guided_registry.layer_indices) != (
            self.report.adapter_layer_indices_sha256
        ):
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Guided SwiGLU layer indices changed."
            )
        before = _unique_parameter_count(self.model)
        for installed in self.inherited_handle._installed.values():
            installed._detach_guided_registry(self.guided_registry)
        if getattr(self.model, self.report.owner_name, None) is self.guided_registry:
            delattr(self.model, self.report.owner_name)
        elif hasattr(self.model, self.report.owner_name):
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Guided SwiGLU owner changed during rollback."
            )
        for name, value in self.registry.named_parameters(recurse=False):
            value.requires_grad_(self._inherited_requires_grad[name])
        if (
            _unique_parameter_count(self.model)
            != before - self.report.guided_adapter_parameter_count
        ):
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Guided-only rollback parameter accounting failed."
            )
        self._state = "rolled-back-guided"
        if validate_inherited_factors:
            self.inherited_handle.assert_live()
        return self.inherited_handle

    def rollback_guided_only(
        self,
    ) -> InstalledDenseSwiGLUProjectionResidualAdapter:
        """Remove the guided product while preserving a valid inherited one."""

        return self._rollback_guided_only(validate_inherited_factors=True)

    def rollback(self) -> None:
        if self._state == "rolled-back":
            return
        if self._state == "baked":
            raise DenseSwiGLUProjectionResidualTopologyError(
                "A published guided bake cannot use live rollback."
            )
        if self._state == "live":
            # Full rejection must be able to dispose both products after an
            # optimizer failure poisoned either registry.  The inherited
            # handle performs its own topology-only, NaN-safe rollback next.
            self._rollback_guided_only(validate_inherited_factors=False)
        if self.inherited_handle.is_applied:
            self.inherited_handle.rollback()
        self._state = "rolled-back"

    @staticmethod
    def _delta_manifest(
        updates: Mapping[tuple[int, SwiGLUProjectionRole], Tensor],
    ) -> tuple[list[dict[str, Any]], str]:
        manifest = [
            {
                "layer": layer,
                "role": role,
                "shape": list(delta.shape),
                "sha256": _tensor_sha256(delta),
            }
            for (layer, role), delta in sorted(updates.items())
        ]
        return manifest, _canonical_sha256(manifest)

    def bake_for_publication(
        self,
    ) -> GuidedDenseSwiGLUProjectionResidualBakeReport:
        """Bake inherited first and guided second into ordinary projections."""

        self.assert_live()
        before_bake = _unique_parameter_count(self.model)
        originals = self.inherited_handle._originals
        inherited_updates: dict[tuple[int, SwiGLUProjectionRole], Tensor] = {}
        guided_updates: dict[tuple[int, SwiGLUProjectionRole], Tensor] = {}
        for key, original in originals.items():
            layer, role = key
            inherited = (
                self.registry.dense_delta(role, layer)
                .detach()
                .to(device=original.weight.device, dtype=original.weight.dtype)
            )
            guided = (
                self.guided_registry.dense_delta(role, layer)
                .detach()
                .to(device=original.weight.device, dtype=original.weight.dtype)
            )
            if tuple(inherited.shape) != tuple(original.weight.shape) or tuple(
                guided.shape
            ) != tuple(original.weight.shape):
                raise DenseSwiGLUProjectionResidualTopologyError(
                    "Materialized guided SwiGLU delta has the wrong shape."
                )
            inherited_updates[key] = inherited
            guided_updates[key] = guided
        expected_updates = 3 * len(self.report.layer_indices)
        if (
            len(inherited_updates) != expected_updates
            or len(guided_updates) != expected_updates
        ):
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Sequential SwiGLU bake did not produce three updates per layer."
            )
        inherited_manifest, inherited_sha256 = self._delta_manifest(inherited_updates)
        guided_manifest, guided_sha256 = self._delta_manifest(guided_updates)
        sequential_sha256 = _canonical_sha256(
            {
                "order": ["inherited", "guided"],
                "inherited": inherited_manifest,
                "guided": guided_manifest,
            }
        )
        published: dict[tuple[int, SwiGLUProjectionRole], nn.Linear] = {}
        try:
            with torch.no_grad():
                for key, original in originals.items():
                    replacement = nn.Linear(
                        original.in_features,
                        original.out_features,
                        bias=original.bias is not None,
                        device=original.weight.device,
                        dtype=original.weight.dtype,
                    )
                    replacement.weight.copy_(original.weight)
                    replacement.weight.add_(inherited_updates[key])
                    replacement.weight.add_(guided_updates[key])
                    replacement.weight.requires_grad_(original.weight.requires_grad)
                    if original.bias is not None:
                        assert replacement.bias is not None
                        replacement.bias.copy_(original.bias)
                        replacement.bias.requires_grad_(original.bias.requires_grad)
                    replacement.train(original.training)
                    published[key] = replacement
            if any(
                not bool(torch.isfinite(value.weight.detach()).all())
                or (
                    value.bias is not None
                    and not bool(torch.isfinite(value.bias.detach()).all())
                )
                for value in published.values()
            ):
                raise DenseSwiGLUProjectionResidualError(
                    "Sequential SwiGLU bake produced a non-finite projection."
                )
            for (layer, role), replacement in published.items():
                setattr(
                    self.mlp_modules[layer],
                    self.report.projection_names[role],
                    replacement,
                )
            if getattr(self.model, self.report.owner_name, None) is not (
                self.guided_registry
            ):
                raise DenseSwiGLUProjectionResidualTopologyError(
                    "Guided registry owner changed during publication."
                )
            delattr(self.model, self.report.owner_name)
            inherited_owner = self.inherited_handle.report.owner_name
            if getattr(self.model, inherited_owner, None) is not self.registry:
                raise DenseSwiGLUProjectionResidualTopologyError(
                    "Inherited registry owner changed during publication."
                )
            delattr(self.model, inherited_owner)
            after = _unique_parameter_count(self.model)
            if after != self.report.model_parameter_count_before:
                raise DenseSwiGLUProjectionResidualAccountingError(
                    "Sequential SwiGLU bake did not restore the native budget."
                )
            if any(
                type(
                    getattr(
                        self.mlp_modules[layer],
                        self.report.projection_names[role],
                    )
                )
                is not nn.Linear
                for layer, role in originals
            ):
                raise DenseSwiGLUProjectionResidualTopologyError(
                    "Sequential SwiGLU publication left a projection wrapper."
                )
        except Exception:
            inherited_owner = self.inherited_handle.report.owner_name
            if getattr(self.model, inherited_owner, None) is not self.registry:
                if hasattr(self.model, inherited_owner):
                    delattr(self.model, inherited_owner)
                self.model.add_module(inherited_owner, self.registry)
            if getattr(self.model, self.report.owner_name, None) is not (
                self.guided_registry
            ):
                if hasattr(self.model, self.report.owner_name):
                    delattr(self.model, self.report.owner_name)
                self.model.add_module(self.report.owner_name, self.guided_registry)
            for (layer, role), installed in self.inherited_handle._installed.items():
                setattr(
                    self.mlp_modules[layer],
                    self.report.projection_names[role],
                    installed,
                )
            raise
        for installed in self.inherited_handle._installed.values():
            installed._detach_guided_registry(self.guided_registry)
        report = GuidedDenseSwiGLUProjectionResidualBakeReport(
            layer_indices=self.report.layer_indices,
            rank=self.report.rank,
            inherited_adapter_parameters_removed=(
                self.report.inherited_adapter_parameter_count
            ),
            guided_adapter_parameters_removed=(
                self.report.guided_adapter_parameter_count
            ),
            adapter_parameters_removed=self.report.adapter_parameter_count,
            model_parameter_count_before_installation=(
                self.report.model_parameter_count_before
            ),
            model_parameter_count_before_bake=before_bake,
            model_parameter_count_after_bake=after,
            inherited_dense_delta_sha256=inherited_sha256,
            guided_dense_delta_sha256=guided_sha256,
            sequential_dense_delta_sha256=sequential_sha256,
            affected_projection_count=(len(inherited_updates) + len(guided_updates)),
        )
        inherited_report = DenseSwiGLUProjectionResidualBakeReport(
            layer_indices=self.inherited_handle.report.layer_indices,
            rank=self.inherited_handle.report.rank,
            adapter_parameters_removed=(
                self.inherited_handle.report.adapter_parameter_count
            ),
            model_parameter_count_before_installation=(
                self.inherited_handle.report.model_parameter_count_before
            ),
            model_parameter_count_before_bake=before_bake,
            model_parameter_count_after_bake=after,
            dense_delta_sha256=inherited_sha256,
            affected_projection_count=len(inherited_updates),
        )
        self.inherited_handle._state = "baked"
        self.inherited_handle._bake_report = inherited_report
        self.inherited_handle._published = published
        self._published = published
        self._bake_report = report
        self._state = "baked"
        return report

    def restore_prepublication_native(
        self,
        *,
        expected_model_parameter_count: int | None = None,
    ) -> None:
        if self._state == "live":
            self.rollback()
            return
        if self._state in {"rolled-back", "rolled-back-guided"}:
            return
        for (layer, role), published in self._published.items():
            if (
                getattr(
                    self.mlp_modules[layer], self.report.projection_names[role], None
                )
                is not published
            ):
                raise DenseSwiGLUProjectionResidualTopologyError(
                    "Published guided SwiGLU ownership changed; restoration unsafe."
                )
        for (layer, role), original in self.inherited_handle._originals.items():
            setattr(
                self.mlp_modules[layer], self.report.projection_names[role], original
            )
        self.inherited_handle._assert_base_unchanged()
        expected = (
            self.inherited_handle.report.model_parameter_count_before
            if expected_model_parameter_count is None
            else _index(
                expected_model_parameter_count,
                "expected_model_parameter_count",
                positive=True,
            )
        )
        if _unique_parameter_count(self.model) != expected:
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Prepublication guided SwiGLU restore changed the native budget."
            )
        self._published.clear()
        self.inherited_handle._published.clear()
        self.inherited_handle._state = "rolled-back"
        self._state = "rolled-back"

    def __enter__(self) -> InstalledGuidedDenseSwiGLUProjectionResidualTranche:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        if self._state == "live":
            self.rollback()


class GuidedDenseSwiGLUProjectionOutputTrainingScope:
    """Freeze every unselected parameter for one guided G/U/D stage."""

    def __init__(
        self, handle: InstalledGuidedDenseSwiGLUProjectionResidualTranche
    ) -> None:
        if not isinstance(handle, InstalledGuidedDenseSwiGLUProjectionResidualTranche):
            raise DenseSwiGLUProjectionResidualTrainingError(
                "Guided training scope requires a live guided SwiGLU handle."
            )
        self.handle = handle
        self._frozen: tuple[
            tuple[nn.Parameter, bool, Tensor | None, int, int], ...
        ] = ()
        self._modes: tuple[tuple[nn.Module, bool], ...] = ()
        self._entered = False

    def __enter__(self) -> tuple[nn.Parameter, ...]:
        if self._entered:
            raise DenseSwiGLUProjectionResidualTrainingError(
                "Guided output training scope cannot be re-entered."
            )
        trainable = self.handle.output_trainable_parameters()
        trainable_ids = {id(value) for value in trainable}
        self._frozen = tuple(
            (
                value,
                bool(value.requires_grad),
                value.grad,
                int(value._version),
                int(value.data_ptr()),
            )
            for value in self.handle.model.parameters()
            if id(value) not in trainable_ids
        )
        self._modes = tuple(
            (module, bool(module.training)) for module in self.handle.model.modules()
        )
        for value, _requires_grad, _gradient, _version, _pointer in self._frozen:
            value.requires_grad_(False)
        self.handle.model.eval()
        self._entered = True
        return trainable

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        if not self._entered:
            return
        for value, requires_grad, gradient, _version, _pointer in self._frozen:
            value.requires_grad_(requires_grad)
            if value.grad is not gradient:
                value.grad = gradient
        for module, training in self._modes:
            module.training = training
        self._entered = False
        if any(
            int(value._version) != version
            or int(value.data_ptr()) != pointer
            or value.grad is not gradient
            or bool(value.requires_grad) != requires_grad
            for value, requires_grad, gradient, version, pointer in self._frozen
        ):
            raise DenseSwiGLUProjectionResidualTrainingError(
                "An unselected parameter changed inside guided SwiGLU training."
            )
        self.handle.assert_live()


class DenseSwiGLUAdapterOnlyOutputTrainingScope:
    """Freeze the target and expose only Gate/Up/Down residual factors."""

    def __init__(self, handle: InstalledDenseSwiGLUProjectionResidualAdapter) -> None:
        self.handle = handle
        self._base: tuple[tuple[nn.Parameter, bool, Tensor | None, int, int], ...] = ()
        self._modes: tuple[tuple[nn.Module, bool], ...] = ()
        self._entered = False

    def __enter__(self) -> tuple[nn.Parameter, ...]:
        if self._entered:
            raise DenseSwiGLUProjectionResidualTrainingError(
                "Output-training scope cannot be re-entered."
            )
        trainable = self.handle.output_trainable_parameters()
        trainable_ids = {id(value) for value in trainable}
        self._base = tuple(
            (
                value,
                value.requires_grad,
                value.grad,
                int(value._version),
                int(value.data_ptr()),
            )
            for value in self.handle.model.parameters()
            if id(value) not in trainable_ids
        )
        self._modes = tuple(
            (module, module.training) for module in self.handle.model.modules()
        )
        for value, *_ in self._base:
            value.requires_grad_(False)
        self.handle.model.eval()
        self._entered = True
        return trainable

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        if not self._entered:
            return
        for value, requires_grad, gradient, _version, _pointer in self._base:
            value.requires_grad_(requires_grad)
            value.grad = gradient
        for module, training in self._modes:
            module.training = training
        self._entered = False
        if any(
            int(value._version) != version
            or int(value.data_ptr()) != pointer
            or value.grad is not gradient
            for value, _requires_grad, gradient, version, pointer in self._base
        ):
            raise DenseSwiGLUProjectionResidualTrainingError(
                "A base parameter changed during residual-only output training."
            )
        self.handle.assert_live()


def install_dense_swiglu_projection_residual_adapter(
    model: nn.Module,
    mlp_modules: Mapping[int, nn.Module] | Sequence[nn.Module],
    *,
    layer_indices: Sequence[int],
    hidden_size: int,
    intermediate_size: int,
    rank: int,
    projection_names: Mapping[str, str] | None = None,
    owner_name: str = "dense_swiglu_projection_residual_registry",
    initialization_seed: int = 0,
) -> InstalledDenseSwiGLUProjectionResidualAdapter:
    """Install role-complete low-rank corrections on a native dense MLP."""

    if not isinstance(model, nn.Module):
        raise DenseSwiGLUProjectionResidualTopologyError(
            "model must be torch.nn.Module."
        )
    if (
        not isinstance(owner_name, str)
        or not owner_name.isidentifier()
        or hasattr(model, owner_name)
    ):
        raise DenseSwiGLUProjectionResidualTopologyError(
            "owner_name must be an unused model attribute identifier."
        )
    selected = tuple(_index(value, "layer_index") for value in layer_indices)
    if not selected or selected != tuple(sorted(set(selected))):
        raise DenseSwiGLUProjectionResidualTopologyError(
            "layer_indices must be nonempty, sorted, and unique."
        )
    modules = (
        dict(mlp_modules)
        if isinstance(mlp_modules, Mapping)
        else dict(enumerate(mlp_modules))
    )
    if any(layer not in modules for layer in selected) or any(
        not isinstance(modules[layer], nn.Module) for layer in selected
    ):
        raise DenseSwiGLUProjectionResidualTopologyError(
            "mlp_modules does not own every selected layer."
        )
    model_modules = {id(module) for module in model.modules()}
    if any(id(modules[layer]) not in model_modules for layer in selected):
        raise DenseSwiGLUProjectionResidualTopologyError(
            "A selected MLP module is not owned by the target model."
        )
    hidden = _index(hidden_size, "hidden_size", positive=True)
    intermediate = _index(intermediate_size, "intermediate_size", positive=True)
    names = _normalized_projection_names(projection_names)
    originals: dict[tuple[int, SwiGLUProjectionRole], nn.Linear] = {}
    expected_shapes = {
        "gate": (intermediate, hidden),
        "up": (intermediate, hidden),
        "down": (hidden, intermediate),
    }
    for layer in selected:
        for role in SparseDenseSwiGLUProjectionResidualRegistry._ROLES:
            projection = getattr(modules[layer], names[role], None)
            if (
                type(projection) is not nn.Linear
                or tuple(projection.weight.shape) != (expected_shapes[role])
            ):
                raise DenseSwiGLUProjectionResidualTopologyError(
                    f"Layer {layer} {role} is not the declared dense projection."
                )
            originals[(layer, role)] = projection
    dtype_devices = {
        (value.weight.dtype, value.weight.device) for value in originals.values()
    }
    if len(dtype_devices) != 1:
        raise DenseSwiGLUProjectionResidualTopologyError(
            "Selected Gate/Up/Down weights must share one dtype and device."
        )
    all_named = tuple(model.named_parameters(remove_duplicate=False))
    for (layer, role), projection in originals.items():
        aliases = [name for name, value in all_named if value is projection.weight]
        if len(aliases) != 1:
            raise DenseSwiGLUProjectionResidualTopologyError(
                f"Layer {layer} {role} weight is aliased; native bake is unsafe."
            )
    dtype, device = next(iter(dtype_devices))
    # These factors are transient optimizer parameters.  FP16 AdamW stores its
    # moments in FP16 as well, where the usual 1e-8 epsilon becomes exact zero
    # and zero/tiny gradients can create NaNs on the first update.  Train the
    # factors in FP32 and cast only their residual output/native bake to the
    # child weight dtype.
    factor_dtype = torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
    registry = SparseDenseSwiGLUProjectionResidualRegistry(
        selected,
        hidden_size=hidden,
        intermediate_size=intermediate,
        rank=rank,
        dtype=factor_dtype,
        device=device,
        initialization_seed=initialization_seed,
    )
    state = _projection_state(originals)
    before = _unique_parameter_count(model)
    installed = {
        key: DenseSwiGLUResidualProjection(
            projection,
            registry,
            layer_index=key[0],
            role=key[1],
        )
        for key, projection in originals.items()
    }
    try:
        model.add_module(owner_name, registry)
        for (layer, role), view in installed.items():
            setattr(modules[layer], names[role], view)
        after = _unique_parameter_count(model)
        accounting = DenseSwiGLUProjectionResidualAccounting.build(
            hidden_size=hidden,
            intermediate_size=intermediate,
            rank=rank,
            site_count=len(selected),
        )
        if (
            not registry.starts_as_exact_zero_correction
            or not registry.input_factors_are_nonzero
            or accounting.adapter_parameters != registry.learned_parameter_count
            or after != before + registry.learned_parameter_count
        ):
            raise DenseSwiGLUProjectionResidualAccountingError(
                "Installed SwiGLU factor accounting or zero identity failed."
            )
        report = DenseSwiGLUProjectionResidualInstallationReport(
            layer_indices=selected,
            hidden_size=hidden,
            intermediate_size=intermediate,
            rank=_index(rank, "rank", positive=True),
            factor_dtype=str(factor_dtype),
            owner_name=owner_name,
            projection_names=names,
            adapter_parameter_count=registry.learned_parameter_count,
            model_parameter_count_before=before,
            model_parameter_count_installed=after,
            base_projection_state_sha256=_projection_state_sha256(state),
            accounting=accounting,
        )
        handle = InstalledDenseSwiGLUProjectionResidualAdapter(
            model=model,
            mlp_modules=modules,
            registry=registry,
            report=report,
            originals=originals,
            installed=installed,
            projection_state_before=state,
        )
        handle.assert_live()
        return handle
    except Exception:
        for (layer, role), original in originals.items():
            setattr(modules[layer], names[role], original)
        if getattr(model, owner_name, None) is registry:
            delattr(model, owner_name)
        raise


def install_guided_dense_swiglu_projection_residual_tranche(
    inherited_handle: InstalledDenseSwiGLUProjectionResidualAdapter,
    *,
    guided_right_rows: Mapping[SwiGLUProjectionRole, Tensor],
    source_basis_tensor_sha256s: Mapping[SwiGLUProjectionRole, str],
    expected_audit_report_sha256: str,
    expected_basis_envelope_file_sha256: str,
    expected_basis_payload_sha256: str,
    expected_basis_manifest_sha256: str,
    owner_name: str = "guided_dense_swiglu_projection_residual_registry",
) -> InstalledGuidedDenseSwiGLUProjectionResidualTranche:
    """Install a second exact-zero behavior-guided Gate/Up/Down product."""

    if not isinstance(inherited_handle, InstalledDenseSwiGLUProjectionResidualAdapter):
        raise DenseSwiGLUProjectionResidualTopologyError(
            "inherited_handle must be a live dense SwiGLU handle."
        )
    inherited_handle.assert_live()
    model = inherited_handle.model
    if (
        not isinstance(owner_name, str)
        or not owner_name.isidentifier()
        or hasattr(model, owner_name)
        or owner_name == inherited_handle.report.owner_name
    ):
        raise DenseSwiGLUProjectionResidualTopologyError(
            "owner_name must be a distinct unused model attribute identifier."
        )
    registry = inherited_handle.registry
    inherited_parameters = tuple(registry.named_parameters(recurse=False))
    if not inherited_parameters:
        raise DenseSwiGLUProjectionResidualTopologyError(
            "The inherited SwiGLU registry has no factors."
        )
    dtype_devices = {
        (value.dtype, value.device) for _name, value in inherited_parameters
    }
    if len(dtype_devices) != 1:
        raise DenseSwiGLUProjectionResidualTopologyError(
            "Inherited SwiGLU factors must share one dtype and device."
        )
    factor_dtype, device = next(iter(dtype_devices))
    if factor_dtype != torch.float32:
        raise DenseSwiGLUProjectionResidualTopologyError(
            "The exact guided SwiGLU tranche requires FP32 temporary factors."
        )
    inherited_factor_sha256 = _swiglu_factor_state_sha256(registry)
    inherited_requires_grad = {
        name: bool(value.requires_grad) for name, value in inherited_parameters
    }
    before_count = _unique_parameter_count(model)
    guided_registry = GuidedSparseDenseSwiGLUProjectionResidualRegistry(
        inherited_handle.report.layer_indices,
        hidden_size=inherited_handle.report.hidden_size,
        intermediate_size=inherited_handle.report.intermediate_size,
        rank=inherited_handle.report.rank,
        guided_right_rows=guided_right_rows,
        source_basis_tensor_sha256s=source_basis_tensor_sha256s,
        audit_report_sha256=_required_sha256(
            expected_audit_report_sha256,
            "expected_audit_report_sha256",
        ),
        basis_envelope_file_sha256=_required_sha256(
            expected_basis_envelope_file_sha256,
            "expected_basis_envelope_file_sha256",
        ),
        basis_payload_sha256=_required_sha256(
            expected_basis_payload_sha256,
            "expected_basis_payload_sha256",
        ),
        basis_manifest_sha256=_required_sha256(
            expected_basis_manifest_sha256,
            "expected_basis_manifest_sha256",
        ),
        dtype=factor_dtype,
        device=device,
    )
    guided_parameter_count = guided_registry.learned_parameter_count
    if guided_parameter_count != registry.learned_parameter_count:
        raise DenseSwiGLUProjectionResidualAccountingError(
            "Guided and inherited SwiGLU products must have equal exact size."
        )
    attached: list[DenseSwiGLUResidualProjection] = []
    try:
        model.add_module(owner_name, guided_registry)
        for installed in inherited_handle._installed.values():
            installed._attach_guided_registry(guided_registry)
            attached.append(installed)
        installed_count = _unique_parameter_count(model)
        if installed_count != before_count + guided_parameter_count:
            raise DenseSwiGLUProjectionResidualAccountingError(
                "Guided SwiGLU installation parameter accounting failed."
            )
        if _swiglu_factor_state_sha256(registry) != inherited_factor_sha256:
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Guided installation changed inherited SwiGLU factors."
            )
        inherited_payload = inherited_handle.report.as_dict()
        report = GuidedDenseSwiGLUProjectionResidualInstallationReport(
            layer_indices=inherited_handle.report.layer_indices,
            hidden_size=inherited_handle.report.hidden_size,
            intermediate_size=inherited_handle.report.intermediate_size,
            rank=inherited_handle.report.rank,
            factor_dtype=str(factor_dtype),
            owner_name=owner_name,
            projection_names=MappingProxyType(
                dict(inherited_handle.report.projection_names)
            ),
            inherited_adapter_parameter_count=(
                inherited_handle.report.adapter_parameter_count
            ),
            guided_adapter_parameter_count=guided_parameter_count,
            adapter_parameter_count=(
                inherited_handle.report.adapter_parameter_count + guided_parameter_count
            ),
            model_parameter_count_before=(
                inherited_handle.report.model_parameter_count_before
            ),
            model_parameter_count_before_guided_installation=before_count,
            model_parameter_count_installed=installed_count,
            adapter_layer_indices_sha256=_tensor_sha256(guided_registry.layer_indices),
            base_projection_state_sha256=(
                inherited_handle.report.base_projection_state_sha256
            ),
            audit_report_sha256=guided_registry.audit_report_sha256,
            basis_envelope_file_sha256=(guided_registry.basis_envelope_file_sha256),
            basis_payload_sha256=guided_registry.basis_payload_sha256,
            basis_manifest_sha256=guided_registry.basis_manifest_sha256,
            source_basis_tensor_sha256s=MappingProxyType(
                dict(guided_registry.source_basis_tensor_sha256s)
            ),
            inherited_installation_sha256=str(inherited_payload["installation_sha256"]),
            inherited_factor_state_sha256_at_install=inherited_factor_sha256,
            inherited_handle_identity=id(inherited_handle),
            inherited_registry_identity=id(registry),
        )
        handle = InstalledGuidedDenseSwiGLUProjectionResidualTranche(
            inherited_handle=inherited_handle,
            guided_registry=guided_registry,
            report=report,
            inherited_requires_grad=inherited_requires_grad,
        )
        handle.assert_live()
        return handle
    except Exception:
        for installed in reversed(attached):
            if installed.guided_registry is guided_registry:
                installed._detach_guided_registry(guided_registry)
        if getattr(model, owner_name, None) is guided_registry:
            delattr(model, owner_name)
        for name, value in inherited_parameters:
            value.requires_grad_(inherited_requires_grad[name])
        if _unique_parameter_count(model) != before_count:
            raise DenseSwiGLUProjectionResidualTopologyError(
                "Failed guided installation could not restore exact accounting."
            )
        raise


__all__ = [
    "DENSE_SWIGLU_PROJECTION_RESIDUAL_ACCOUNTING_SCHEMA",
    "DENSE_SWIGLU_PROJECTION_RESIDUAL_BAKE_SCHEMA",
    "DENSE_SWIGLU_PROJECTION_RESIDUAL_INSTALLATION_SCHEMA",
    "GUIDED_DENSE_SWIGLU_PROJECTION_RESIDUAL_BAKE_SCHEMA",
    "GUIDED_DENSE_SWIGLU_PROJECTION_RESIDUAL_INSTALLATION_SCHEMA",
    "GUIDED_DENSE_SWIGLU_PROJECTION_RESIDUAL_REGISTRY_SCHEMA",
    "DenseSwiGLUAdapterOnlyOutputTrainingScope",
    "DenseSwiGLUProjectionResidualAccounting",
    "DenseSwiGLUProjectionResidualAccountingError",
    "DenseSwiGLUProjectionResidualBakeReport",
    "DenseSwiGLUProjectionResidualError",
    "DenseSwiGLUProjectionResidualInstallationReport",
    "DenseSwiGLUProjectionResidualTopologyError",
    "DenseSwiGLUProjectionResidualTrainingError",
    "DenseSwiGLUResidualProjection",
    "GuidedDenseSwiGLUProjectionOutputTrainingScope",
    "GuidedDenseSwiGLUProjectionResidualBakeReport",
    "GuidedDenseSwiGLUProjectionResidualInstallationReport",
    "GuidedSparseDenseSwiGLUProjectionResidualRegistry",
    "GuidedSwiGLUTrainingStage",
    "InstalledDenseSwiGLUProjectionResidualAdapter",
    "InstalledGuidedDenseSwiGLUProjectionResidualTranche",
    "SparseDenseSwiGLUProjectionResidualRegistry",
    "SwiGLUProjectionRole",
    "install_dense_swiglu_projection_residual_adapter",
    "install_guided_dense_swiglu_projection_residual_tranche",
]
