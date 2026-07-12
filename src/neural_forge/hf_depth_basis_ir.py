"""Runtime depth-basis IR for Llama and Qwen3 decoder projections.

Each logical projection remains at its original decoder path.  Parameterless
fixed-layer views dispatch to one model-owned :class:`DepthBasisLinear` per
logical role, so PyTorch registers every physical parameter exactly once.
"""

from __future__ import annotations

import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from neural_forge.hf_arch_layout import (
    DecoderLayout,
    UnsupportedDecoderLayoutError,
    build_decoder_layout,
    resolve_module,
)
from neural_forge.hf_basis_linear import (
    DEFAULT_FEATURE_CHUNK_SIZE,
    DepthBasisLinear,
    DepthBasisReport,
    factorize_linears,
    materialize_linears,
)


PROJECTION_ROLES = ("q", "k", "v", "o", "gate", "up", "down")
_REGISTRY_NAME = "_depth_basis_ir"


class HFDepthBasisIRError(ValueError):
    """Base class for runtime depth-basis IR failures."""


class HFDepthBasisCompilationError(HFDepthBasisIRError):
    """Raised when compilation or materialization cannot commit atomically."""


@dataclass(frozen=True, slots=True)
class DepthBasisSelector:
    """At most one per-role depth-basis selector."""

    rank: int | None = None
    energy: float | None = None
    parameter_budget: int | None = None


@dataclass(frozen=True, slots=True)
class DepthBasisViewAlias:
    """A logical projection path backed by one canonical representation."""

    role: str
    layer_index: int
    module_path: str
    representation_path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "layer_index": self.layer_index,
            "module_path": self.module_path,
            "representation_path": self.representation_path,
        }


@dataclass(frozen=True, slots=True)
class DepthBasisRoleReport:
    """Physical and logical accounting for one projection role."""

    role: str
    paths: tuple[str, ...]
    representation_path: str
    state_dict_keys: tuple[str, ...]
    basis: DepthBasisReport

    @property
    def logical_layers(self) -> int:
        return self.basis.logical_layers

    @property
    def rank(self) -> int:
        return self.basis.rank

    @property
    def logical_parameter_count(self) -> int:
        return self.basis.logical_parameter_count

    @property
    def unique_parameter_count(self) -> int:
        return self.basis.unique_parameter_count

    @property
    def parameters_saved(self) -> int:
        return self.basis.parameters_saved

    def as_dict(self) -> dict[str, Any]:
        result = self.basis.as_dict()
        result.update(
            {
                "role": self.role,
                "paths": list(self.paths),
                "representation_path": self.representation_path,
                "state_dict_keys": list(self.state_dict_keys),
            }
        )
        return result

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


@dataclass(frozen=True, slots=True)
class HFDepthBasisIRReport:
    """Whole-model accounting and alias evidence for a compiled IR."""

    family: str
    execution_depth: int
    source_logical_parameters: int
    source_unique_parameters: int
    compiled_logical_parameters: int
    compiled_unique_parameters: int
    selected_logical_parameters: int
    selected_unique_parameters: int
    roles: tuple[DepthBasisRoleReport, ...]
    aliases: tuple[DepthBasisViewAlias, ...]

    @property
    def unique_parameters_saved(self) -> int:
        return self.source_unique_parameters - self.compiled_unique_parameters

    @property
    def logical_parameter_count(self) -> int:
        return self.compiled_logical_parameters

    @property
    def unique_parameter_count(self) -> int:
        return self.compiled_unique_parameters

    @property
    def all_logical_layers_executed(self) -> bool:
        return all(role.logical_layers == self.execution_depth for role in self.roles)

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "execution_depth": self.execution_depth,
            "source_logical_parameter_count": self.source_logical_parameters,
            "source_unique_parameter_count": self.source_unique_parameters,
            "compiled_logical_parameter_count": self.compiled_logical_parameters,
            "compiled_unique_parameter_count": self.compiled_unique_parameters,
            "selected_logical_parameter_count": self.selected_logical_parameters,
            "selected_unique_parameter_count": self.selected_unique_parameters,
            "unique_parameters_saved": self.unique_parameters_saved,
            "all_logical_layers_executed": self.all_logical_layers_executed,
            "roles": {role.role: role.as_dict() for role in self.roles},
            "aliases": [alias.as_dict() for alias in self.aliases],
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


class FixedLayerDepthBasisLinear(nn.Module):
    """Parameterless ``nn.Linear``-signature view of one logical depth row."""

    def __init__(self, representation: DepthBasisLinear, layer_index: int) -> None:
        super().__init__()
        if not isinstance(representation, DepthBasisLinear):
            raise HFDepthBasisIRError("representation must be a DepthBasisLinear.")
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or not 0 <= layer_index < representation.logical_layers
        ):
            raise HFDepthBasisIRError(
                "layer_index must identify a logical layer in the representation."
            )
        object.__setattr__(self, "_representation_ref", weakref.ref(representation))
        self.layer_index = layer_index
        self.in_features = representation.in_features
        self.out_features = representation.out_features
        self.train(representation.training)

    @property
    def representation(self) -> DepthBasisLinear:
        representation = self._representation_ref()
        if representation is None:
            raise HFDepthBasisCompilationError(
                "The shared depth-basis representation is no longer registered."
            )
        return representation

    @property
    def has_bias(self) -> bool:
        return self.representation.has_bias

    def forward(self, input: Tensor) -> Tensor:  # noqa: A002
        return self.representation(input, self.layer_index)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"layer_index={self.layer_index}, bias={self.has_bias}"
        )


class DepthBasisRegistry(nn.Module):
    """Canonical model-owned storage for role representations."""

    def __init__(self, representations: Mapping[str, DepthBasisLinear]) -> None:
        super().__init__()
        if tuple(representations) != PROJECTION_ROLES:
            raise HFDepthBasisIRError(
                f"representations must contain roles in order {PROJECTION_ROLES!r}."
            )
        self.representations = nn.ModuleDict(dict(representations))


@dataclass(slots=True)
class CompiledHFDepthBasisIR:
    """Handle for an in-place compiled runtime IR."""

    model: nn.Module
    layout: DecoderLayout
    registry: DepthBasisRegistry
    role_paths: dict[str, tuple[str, ...]]
    report: HFDepthBasisIRReport
    materialized: bool = False

    @property
    def representations(self) -> Mapping[str, DepthBasisLinear]:
        return self.registry.representations

    @property
    def role_reports(self) -> dict[str, DepthBasisRoleReport]:
        return {role.role: role for role in self.report.roles}

    @property
    def logical_parameter_count(self) -> int:
        return self.report.logical_parameter_count

    @property
    def unique_parameter_count(self) -> int:
        return self.report.unique_parameter_count

    def as_dict(self) -> dict[str, Any]:
        result = self.report.as_dict()
        result["materialized"] = self.materialized
        return result

    def materialize(self) -> nn.Module:
        return materialize_hf_depth_basis_ir(self)


def _projection_paths(layout: DecoderLayout) -> dict[str, tuple[str, ...]]:
    if layout.family not in {"llama", "qwen3"}:
        raise UnsupportedDecoderLayoutError(
            "Depth-basis projection IR supports only split-projection Llama and "
            f"Qwen3 decoders, got {layout.family!r}."
        )
    paths: dict[str, list[str]] = {role: [] for role in PROJECTION_ROLES}
    for block in layout.blocks:
        attention = block.attention
        mlp = block.mlp
        if attention.projection_style != "split_qkv" or mlp.kind != "gated":
            raise UnsupportedDecoderLayoutError(
                f"Unsupported {layout.family} projection topology."
            )
        required = {
            "q": attention.query_path,
            "k": attention.key_path,
            "v": attention.value_path,
            "o": attention.output_path,
            "gate": mlp.gate_projection_path,
            "up": mlp.up_projection_path,
            "down": mlp.down_projection_path,
        }
        for role, path in required.items():
            if path is None:
                raise UnsupportedDecoderLayoutError(
                    f"Required projection role {role!r} has no explicit layout path."
                )
            paths[role].append(path)
    return {role: tuple(paths[role]) for role in PROJECTION_ROLES}


def _linear_groups(
    model: nn.Module,
    layout: DecoderLayout,
    role_paths: Mapping[str, Sequence[str]],
) -> dict[str, tuple[nn.Linear, ...]]:
    groups: dict[str, tuple[nn.Linear, ...]] = {}
    for role in PROJECTION_ROLES:
        modules = tuple(
            resolve_module(model, path, family=layout.family)
            for path in role_paths[role]
        )
        if any(type(module) is not nn.Linear for module in modules):
            actual = [type(module).__name__ for module in modules]
            raise UnsupportedDecoderLayoutError(
                f"Projection role {role!r} must contain only exact nn.Linear "
                f"operators, got {actual!r}."
            )
        linears = tuple(modules)
        first = linears[0]
        topology = (
            first.in_features,
            first.out_features,
            first.bias is not None,
            first.weight.dtype,
            first.weight.device,
            first.training,
        )
        for linear in linears[1:]:
            candidate = (
                linear.in_features,
                linear.out_features,
                linear.bias is not None,
                linear.weight.dtype,
                linear.weight.device,
                linear.training,
            )
            if candidate != topology:
                raise UnsupportedDecoderLayoutError(
                    f"Projection role {role!r} is not same-shaped with one bias, "
                    "dtype, device, and training-mode topology across depth."
                )
        groups[role] = linears
    return groups


def _selector_from_value(role: str, value: Any) -> DepthBasisSelector:
    if isinstance(value, DepthBasisSelector):
        return value
    if not isinstance(value, Mapping):
        raise HFDepthBasisIRError(
            f"role_specs[{role!r}] must be a DepthBasisSelector or mapping."
        )
    unknown = set(value) - {"rank", "energy", "parameter_budget"}
    if unknown:
        raise HFDepthBasisIRError(
            f"role_specs[{role!r}] has unsupported keys {sorted(unknown)!r}."
        )
    return DepthBasisSelector(
        rank=value.get("rank"),
        energy=value.get("energy"),
        parameter_budget=value.get("parameter_budget"),
    )


def _selectors(
    role_specs: Mapping[str, DepthBasisSelector | Mapping[str, Any]] | None,
    role_ranks: Mapping[str, int] | None,
    role_energies: Mapping[str, float] | None,
    role_parameter_budgets: Mapping[str, int] | None,
) -> dict[str, DepthBasisSelector]:
    result = {role: DepthBasisSelector() for role in PROJECTION_ROLES}
    assigned: set[str] = set()
    sources: tuple[tuple[str, Mapping[str, Any] | None], ...] = (
        ("role_specs", role_specs),
        ("role_ranks", role_ranks),
        ("role_energies", role_energies),
        ("role_parameter_budgets", role_parameter_budgets),
    )
    for source_name, values in sources:
        if values is None:
            continue
        if not isinstance(values, Mapping):
            raise HFDepthBasisIRError(f"{source_name} must be a mapping by role.")
        unknown = set(values) - set(PROJECTION_ROLES)
        if unknown:
            raise HFDepthBasisIRError(
                f"{source_name} has unsupported roles {sorted(unknown)!r}."
            )
        overlap = assigned.intersection(values)
        if overlap:
            raise HFDepthBasisIRError(
                f"Roles {sorted(overlap)!r} have more than one selector."
            )
        for role, value in values.items():
            if source_name == "role_specs":
                result[role] = _selector_from_value(role, value)
            elif source_name == "role_ranks":
                result[role] = DepthBasisSelector(rank=value)
            elif source_name == "role_energies":
                result[role] = DepthBasisSelector(energy=value)
            else:
                result[role] = DepthBasisSelector(parameter_budget=value)
        assigned.update(values)
    return result


def _parameter_counts(model: nn.Module) -> tuple[int, int]:
    logical = 0
    unique: dict[int, nn.Parameter] = {}
    for module in model.modules():
        for parameter in module._parameters.values():
            if parameter is not None:
                logical += parameter.numel()
                unique[id(parameter)] = parameter
    return logical, sum(parameter.numel() for parameter in unique.values())


def _set_module(model: nn.Module, path: str, module: nn.Module) -> None:
    owner_path, separator, name = path.rpartition(".")
    owner = resolve_module(model, owner_path) if separator else model
    if not isinstance(owner, nn.Module):
        raise HFDepthBasisCompilationError(f"Owner of {path!r} is not a module.")
    setattr(owner, name, module)


def _remove_registry(model: nn.Module, registry: DepthBasisRegistry) -> None:
    if getattr(model, _REGISTRY_NAME, None) is registry:
        delattr(model, _REGISTRY_NAME)


def _restore_modules(
    model: nn.Module, captured: Sequence[tuple[str, nn.Module]]
) -> None:
    for path, module in captured:
        _set_module(model, path, module)


def compile_hf_depth_basis_ir(
    model: nn.Module,
    *,
    role_specs: Mapping[str, DepthBasisSelector | Mapping[str, Any]] | None = None,
    role_ranks: Mapping[str, int] | None = None,
    role_energies: Mapping[str, float] | None = None,
    role_parameter_budgets: Mapping[str, int] | None = None,
    feature_chunk_size: int = DEFAULT_FEATURE_CHUNK_SIZE,
) -> CompiledHFDepthBasisIR:
    """Compile all Llama/Qwen3 decoder projection roles in place atomically."""

    if not isinstance(model, nn.Module):
        raise HFDepthBasisIRError("model must be a torch.nn.Module.")
    if hasattr(model, _REGISTRY_NAME):
        raise HFDepthBasisIRError("model already has a depth-basis IR registry.")
    layout = build_decoder_layout(model)
    role_paths = _projection_paths(layout)
    groups = _linear_groups(model, layout, role_paths)
    selectors = _selectors(
        role_specs,
        role_ranks,
        role_energies,
        role_parameter_budgets,
    )
    source_logical, source_unique = _parameter_counts(model)

    representations: dict[str, DepthBasisLinear] = {}
    for role in PROJECTION_ROLES:
        selector = selectors[role]
        representation = factorize_linears(
            groups[role],
            rank=selector.rank,
            energy=selector.energy,
            parameter_budget=selector.parameter_budget,
            feature_chunk_size=feature_chunk_size,
        )
        representations[role] = representation

    registry = DepthBasisRegistry(representations)
    registry.training = model.training
    captured = tuple(
        (path, groups[role][layer_index])
        for role in PROJECTION_ROLES
        for layer_index, path in enumerate(role_paths[role])
    )
    registry_attached = False
    try:
        setattr(model, _REGISTRY_NAME, registry)
        registry_attached = True
        for role in PROJECTION_ROLES:
            representation = representations[role]
            for layer_index, path in enumerate(role_paths[role]):
                view = FixedLayerDepthBasisLinear(representation, layer_index)
                view.train(groups[role][layer_index].training)
                _set_module(model, path, view)

        for role in PROJECTION_ROLES:
            representation = representations[role]
            for layer_index, path in enumerate(role_paths[role]):
                view = resolve_module(model, path, family=layout.family)
                if (
                    not isinstance(view, FixedLayerDepthBasisLinear)
                    or view.representation is not representation
                    or view.layer_index != layer_index
                ):
                    raise HFDepthBasisCompilationError(
                        f"Projection view validation failed at {path!r}."
                    )
        compiled_logical_slots, compiled_unique = _parameter_counts(model)
        selected_logical = sum(
            representations[role].report.logical_parameter_count
            for role in PROJECTION_ROLES
            if representations[role].report is not None
        )
        selected_unique = sum(
            sum(parameter.numel() for parameter in representations[role].parameters())
            for role in PROJECTION_ROLES
        )
        expected_unique = source_unique - selected_logical + selected_unique
        if compiled_unique != expected_unique:
            raise HFDepthBasisCompilationError(
                "Compiled unique parameter count is inconsistent with selected "
                f"roles: expected {expected_unique}, got {compiled_unique}."
            )
        expected_slots = source_logical - selected_logical + selected_unique
        if compiled_logical_slots != expected_slots:
            raise HFDepthBasisCompilationError(
                "Compiled registered parameter slots are inconsistent with the IR."
            )
    except Exception:
        _restore_modules(model, captured)
        if registry_attached:
            _remove_registry(model, registry)
        raise

    role_reports: list[DepthBasisRoleReport] = []
    aliases: list[DepthBasisViewAlias] = []
    for role in PROJECTION_ROLES:
        representation = representations[role]
        basis_report = representation.report
        if basis_report is None:  # pragma: no cover - factorizer always reports
            _restore_modules(model, captured)
            _remove_registry(model, registry)
            raise HFDepthBasisCompilationError(f"Role {role!r} has no basis report.")
        representation_path = f"{_REGISTRY_NAME}.representations.{role}"
        state_keys = tuple(
            f"{representation_path}.{key}" for key in representation.state_dict()
        )
        role_reports.append(
            DepthBasisRoleReport(
                role=role,
                paths=role_paths[role],
                representation_path=representation_path,
                state_dict_keys=state_keys,
                basis=basis_report,
            )
        )
        aliases.extend(
            DepthBasisViewAlias(
                role=role,
                layer_index=layer_index,
                module_path=path,
                representation_path=representation_path,
            )
            for layer_index, path in enumerate(role_paths[role])
        )
    report = HFDepthBasisIRReport(
        family=layout.family,
        execution_depth=layout.layer_count,
        source_logical_parameters=source_logical,
        source_unique_parameters=source_unique,
        compiled_logical_parameters=source_logical,
        compiled_unique_parameters=compiled_unique,
        selected_logical_parameters=selected_logical,
        selected_unique_parameters=selected_unique,
        roles=tuple(role_reports),
        aliases=tuple(aliases),
    )
    return CompiledHFDepthBasisIR(
        model=model,
        layout=layout,
        registry=registry,
        role_paths=role_paths,
        report=report,
    )


def materialize_hf_depth_basis_ir(compiled: CompiledHFDepthBasisIR) -> nn.Module:
    """Replace every view with an independent ``nn.Linear`` atomically."""

    if not isinstance(compiled, CompiledHFDepthBasisIR):
        raise HFDepthBasisIRError("compiled must be a CompiledHFDepthBasisIR.")
    if compiled.materialized:
        raise HFDepthBasisIRError("This depth-basis IR is already materialized.")
    if getattr(compiled.model, _REGISTRY_NAME, None) is not compiled.registry:
        raise HFDepthBasisCompilationError(
            "The model no longer owns the compiled depth-basis registry."
        )

    captured: list[tuple[str, nn.Module]] = []
    dense: dict[str, list[nn.Linear]] = {}
    for role in PROJECTION_ROLES:
        representation = compiled.registry.representations[role]
        dense[role] = materialize_linears(representation)
        for path in compiled.role_paths[role]:
            current = resolve_module(
                compiled.model, path, family=compiled.layout.family
            )
            if not isinstance(current, FixedLayerDepthBasisLinear):
                raise HFDepthBasisCompilationError(
                    f"Expected a fixed-layer view at {path!r}."
                )
            captured.append((path, current))

    try:
        for role in PROJECTION_ROLES:
            for path, linear in zip(
                compiled.role_paths[role], dense[role], strict=True
            ):
                _set_module(compiled.model, path, linear)
        _remove_registry(compiled.model, compiled.registry)
        _, unique = _parameter_counts(compiled.model)
        if unique != compiled.report.source_unique_parameters:
            raise HFDepthBasisCompilationError(
                "Materialization did not restore the source unique parameter count: "
                f"expected {compiled.report.source_unique_parameters}, got {unique}."
            )
        for role in PROJECTION_ROLES:
            modules = [
                resolve_module(compiled.model, path, family=compiled.layout.family)
                for path in compiled.role_paths[role]
            ]
            if any(type(module) is not nn.Linear for module in modules):
                raise HFDepthBasisCompilationError(
                    f"Role {role!r} did not materialize to exact nn.Linear modules."
                )
            parameter_ids = [
                id(parameter) for module in modules for parameter in module.parameters()
            ]
            if len(parameter_ids) != len(set(parameter_ids)):
                raise HFDepthBasisCompilationError(
                    f"Role {role!r} materialization left parameter aliases."
                )
    except Exception:
        if not hasattr(compiled.model, _REGISTRY_NAME):
            setattr(compiled.model, _REGISTRY_NAME, compiled.registry)
        _restore_modules(compiled.model, captured)
        raise

    compiled.materialized = True
    return compiled.model


compile_depth_basis_ir = compile_hf_depth_basis_ir
materialize_depth_basis_ir = materialize_hf_depth_basis_ir


__all__ = [
    "PROJECTION_ROLES",
    "CompiledHFDepthBasisIR",
    "DepthBasisRegistry",
    "DepthBasisRoleReport",
    "DepthBasisSelector",
    "DepthBasisViewAlias",
    "FixedLayerDepthBasisLinear",
    "HFDepthBasisCompilationError",
    "HFDepthBasisIRError",
    "HFDepthBasisIRReport",
    "compile_depth_basis_ir",
    "compile_hf_depth_basis_ir",
    "materialize_depth_basis_ir",
    "materialize_hf_depth_basis_ir",
]
