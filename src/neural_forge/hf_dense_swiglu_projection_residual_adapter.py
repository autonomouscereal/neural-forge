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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


SwiGLUProjectionRole = Literal["gate", "up", "down"]

DENSE_SWIGLU_PROJECTION_RESIDUAL_ACCOUNTING_SCHEMA = (
    "neural-forge-dense-swiglu-projection-residual-accounting-v1"
)
DENSE_SWIGLU_PROJECTION_RESIDUAL_INSTALLATION_SCHEMA = (
    "neural-forge-dense-swiglu-projection-residual-installation-v2"
)
DENSE_SWIGLU_PROJECTION_RESIDUAL_BAKE_SCHEMA = (
    "neural-forge-dense-swiglu-projection-residual-bake-v1"
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

    def assert_finite(self) -> None:
        if any(
            not bool(torch.isfinite(value.detach()).all())
            for value in self.parameters()
        ):
            raise DenseSwiGLUProjectionResidualError(
                "SwiGLU projection factors contain NaN or infinity."
            )

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
        self.assert_finite()
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
        return base + correction


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


__all__ = [
    "DENSE_SWIGLU_PROJECTION_RESIDUAL_ACCOUNTING_SCHEMA",
    "DENSE_SWIGLU_PROJECTION_RESIDUAL_BAKE_SCHEMA",
    "DENSE_SWIGLU_PROJECTION_RESIDUAL_INSTALLATION_SCHEMA",
    "DenseSwiGLUAdapterOnlyOutputTrainingScope",
    "DenseSwiGLUProjectionResidualAccounting",
    "DenseSwiGLUProjectionResidualAccountingError",
    "DenseSwiGLUProjectionResidualBakeReport",
    "DenseSwiGLUProjectionResidualError",
    "DenseSwiGLUProjectionResidualInstallationReport",
    "DenseSwiGLUProjectionResidualTopologyError",
    "DenseSwiGLUProjectionResidualTrainingError",
    "DenseSwiGLUResidualProjection",
    "InstalledDenseSwiGLUProjectionResidualAdapter",
    "SparseDenseSwiGLUProjectionResidualRegistry",
    "SwiGLUProjectionRole",
    "install_dense_swiglu_projection_residual_adapter",
]
