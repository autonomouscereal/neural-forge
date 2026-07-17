"""Reusable whole-model LFM2 CPCF compiler.

The public operation in this module owns the complete in-memory fold after a
dense target and aligned cached phases have been created.  It fits the tied
interface and independent boundary coordinates, installs every macrocell in
strict order on actual folded-upstream states, fits the final target-coordinate
RMSNorm, and emits a sealed parameter-count receipt.

This module is intentionally independent of experiment scripts, checkpoint
paths, model hubs, and parent execution.  It can compile any representable
parameter budget.  A caller may attach a promotion ratio (for example 100x)
and optionally require it, but lower-ratio diagnostic controls remain valid
fold requests and are explicitly marked non-promotable rather than being
misrepresented as the project goal.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_fold_executor import fingerprint_model_state
from neural_forge.hf_closed_form_causal_circuit_fold import (
    fit_rmsnorm_diagonal,
)
from neural_forge.hf_lfm2_cpcf_boundary_bundle import (
    Lfm2CpcfBoundaryBundle,
    fit_lfm2_cpcf_boundary_bundle,
)
from neural_forge.hf_lfm2_cpcf_layer_compiler import (
    Lfm2CpcfCompiledLayerResult,
    compile_and_install_lfm2_cpcf_layer,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import (
    Lfm2CpcfPhaseStateBank,
    Lfm2CpcfTargetPhase,
    install_lfm2_cpcf_neutral_norms,
    install_lfm2_cpcf_tied_interface,
    seed_lfm2_cpcf_phase_states,
)
from neural_forge.hf_lfm2_cpcf_short_conv_fold import (
    Lfm2CpcfShortConvGatePolicy,
)


LFM2_CPCF_MODEL_FOLD_SCHEMA = "neural-forge-lfm2-cpcf-model-fold-v1"
LFM2_CPCF_PARAMETER_RECEIPT_SCHEMA = (
    "neural-forge-lfm2-cpcf-parameter-receipt-v1"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class Lfm2CpcfModelFoldError(ValueError):
    """The requested complete CPCF model fold is invalid."""


class Lfm2CpcfModelFoldInstallError(RuntimeError):
    """A complete model fold could not be installed honestly."""


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for LFM2 CPCF model folding."
        ) from exc
    return torch


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
        raise Lfm2CpcfModelFoldError(
            "Model-fold metadata must be finite canonical JSON."
        ) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2CpcfModelFoldError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CpcfModelFoldError(
            f"{name} must be a positive integer."
        )
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2CpcfModelFoldError(
            f"{name} must be finite and nonnegative."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CpcfModelFoldError(
            f"{name} must be finite and nonnegative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2CpcfModelFoldError(
            f"{name} must be finite and nonnegative."
        )
    return result


def _fraction(value: Any, name: str) -> float:
    result = _nonnegative_float(value, name)
    if result > 1.0:
        raise Lfm2CpcfModelFoldError(f"{name} must be within [0, 1].")
    return result


def _model_core(model: Any, name: str) -> Any:
    config = getattr(model, "config", None)
    core = getattr(model, "model", None)
    layers = getattr(core, "layers", None)
    if (
        getattr(config, "model_type", None) != "lfm2"
        or core is None
        or layers is None
    ):
        raise Lfm2CpcfModelFoldError(
            f"{name} must be a materialized LFM2 causal language model."
        )
    return core


def independent_parameter_inventory(model: Any) -> Mapping[str, Any]:
    """Count learned scalars by materialized storage identity.

    This deliberately does not trust a topology planner, config declaration,
    checkpoint byte size, quantization metadata, or pruning mask.
    """

    try:
        named = tuple(model.named_parameters(remove_duplicate=False))
    except TypeError:  # pragma: no cover - older Torch compatibility
        named = tuple(model.named_parameters())
    groups: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    raw_numel = 0
    for name, parameter in named:
        if parameter.device.type == "meta":
            raise Lfm2CpcfModelFoldError(
                f"Parameter {name!r} is not materialized."
            )
        raw_numel += int(parameter.numel())
        key = (
            int(parameter.untyped_storage().data_ptr()),
            int(parameter.storage_offset()),
            int(parameter.numel()),
            int(parameter.element_size()),
        )
        group = groups.setdefault(
            key,
            {
                "numel": int(parameter.numel()),
                "shape": [int(size) for size in parameter.shape],
                "dtype": str(parameter.dtype).removeprefix("torch."),
                "names": [],
            },
        )
        group["names"].append(str(name))
    if not groups:
        raise Lfm2CpcfModelFoldError(
            "Model exposes no materialized learned parameters."
        )
    unique_numel = sum(int(group["numel"]) for group in groups.values())
    iterator_numel = sum(
        int(parameter.numel()) for parameter in model.parameters()
    )
    aliases = sorted(
        (
            {
                "names": sorted(group["names"]),
                "numel": group["numel"],
                "shape": group["shape"],
                "dtype": group["dtype"],
            }
            for group in groups.values()
            if len(group["names"]) > 1
        ),
        key=lambda value: value["names"],
    )
    body = {
        "method": (
            "unique_materialized_parameter_storage_offset_numel_identity"
        ),
        "unique_parameters": unique_numel,
        "raw_named_parameter_occurrences": len(named),
        "raw_named_parameter_numel": raw_numel,
        "parameter_iterator_unique_numel": iterator_numel,
        "independent_methods_agree": unique_numel == iterator_numel,
        "alias_group_count": len(aliases),
        "alias_groups": aliases,
        "checkpoint_byte_size_used": False,
        "quantization_adjustment_used": False,
        "pruning_mask_adjustment_used": False,
    }
    if not body["independent_methods_agree"]:
        raise Lfm2CpcfModelFoldError(
            "Independent parameter-count methods disagree."
        )
    return MappingProxyType(
        {**body, "inventory_sha256": _sha(body)}
    )


def audit_lfm2_cpcf_parameter_compaction(
    source_model: Any,
    target_model: Any,
    *,
    planned_target_unique_parameters: int,
    promotion_compression_ratio: float = 1.0,
) -> Mapping[str, Any]:
    """Measure real parameter compaction without restricting valid budgets."""

    planned = _positive_int(
        planned_target_unique_parameters,
        "planned_target_unique_parameters",
    )
    promotion_floor = _nonnegative_float(
        promotion_compression_ratio,
        "promotion_compression_ratio",
    )
    if promotion_floor <= 0.0:
        raise Lfm2CpcfModelFoldError(
            "promotion_compression_ratio must be positive."
        )
    source = independent_parameter_inventory(source_model)
    target = independent_parameter_inventory(target_model)
    source_count = int(source["unique_parameters"])
    target_count = int(target["unique_parameters"])
    if target_count != planned:
        raise Lfm2CpcfModelFoldError(
            "Independent target count differs from the catalog plan."
        )
    if not 0 < target_count < source_count:
        raise Lfm2CpcfModelFoldError(
            "Target topology does not compact the source parameter count."
        )
    ratio = source_count / target_count
    body = {
        "schema": LFM2_CPCF_PARAMETER_RECEIPT_SCHEMA,
        "source_inventory": dict(source),
        "target_inventory": dict(target),
        "source_unique_parameters": source_count,
        "target_unique_parameters": target_count,
        "planned_target_unique_parameters": planned,
        "parameters_removed_from_topology": source_count - target_count,
        "realized_compression_ratio": ratio,
        "promotion_compression_ratio": promotion_floor,
        "promotion_ratio_met": ratio >= promotion_floor,
        "valid_lower_ratio_diagnostic_fold": ratio < promotion_floor,
        "parameter_count_compaction_only": True,
        "checkpoint_byte_ratio_used": False,
        "quantization_used": False,
        "pruning_used": False,
    }
    return MappingProxyType({**body, "receipt_sha256": _sha(body)})


def _blank_receipt(
    value: Mapping[str, Any],
    *,
    target_unique_parameters: int,
) -> Mapping[str, Any]:
    required_true = (
        "compiler_created_blank",
        "all_parameters_zero",
        "dense_materialized_parameters",
        "topology_provenance_verified",
    )
    if (
        not isinstance(value, Mapping)
        or any(value.get(name) is not True for name in required_true)
        or value.get("weights_installed") is not False
        or value.get("training_used") is not False
        or value.get("quantization_used") is not False
        or value.get("pruning_used") is not False
        or int(value.get("target_unique_parameters", -1))
        != target_unique_parameters
    ):
        raise Lfm2CpcfModelFoldError(
            "Target lacks an authenticated compiler-created dense blank receipt."
        )
    report = dict(value)
    report_sha = _digest(
        report.pop("report_sha256", None),
        "blank_target_report.report_sha256",
    )
    if _sha(report) != report_sha:
        raise Lfm2CpcfModelFoldError(
            "Blank-target report seal is invalid."
        )
    return MappingProxyType({**report, "report_sha256": report_sha})


def _assert_target_is_zero_blank(target_model: Any) -> None:
    torch = _torch()
    for name, parameter in target_model.named_parameters():
        if not bool(torch.count_nonzero(parameter.detach()).item() == 0):
            raise Lfm2CpcfModelFoldError(
                f"Target parameter {name!r} is not blank before compilation."
            )


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(size) for size in tensor.shape],
    }
    digest = hashlib.sha256(_canonical_json(descriptor))
    digest.update(
        tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _ModelSnapshot:
    tensor_values: Mapping[str, Any]
    tensor_identity: Mapping[str, tuple[int, int, tuple[int, ...], Any]]
    parameter_gradients: Mapping[
        str,
        tuple[bool, Any | None, Any | None],
    ]
    module_modes: tuple[tuple[str, bool], ...]
    state_fingerprint: str


def _snapshot_model(model: Any) -> _ModelSnapshot:
    state = model.state_dict(keep_vars=True)
    tensor_values = {
        str(name): value.detach().to(device="cpu").contiguous().clone()
        for name, value in state.items()
    }
    tensor_identity = {
        str(name): (
            id(value),
            int(value.data_ptr()),
            tuple(int(size) for size in value.shape),
            value.dtype,
        )
        for name, value in state.items()
    }
    gradients = {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        gradients[str(name)] = (
            bool(parameter.requires_grad),
            gradient,
            (
                None
                if gradient is None
                else gradient.detach().to(device="cpu").contiguous().clone()
            ),
        )
    return _ModelSnapshot(
        tensor_values=MappingProxyType(tensor_values),
        tensor_identity=MappingProxyType(tensor_identity),
        parameter_gradients=MappingProxyType(gradients),
        module_modes=tuple(
            (str(name), bool(module.training))
            for name, module in model.named_modules()
        ),
        state_fingerprint=fingerprint_model_state(model),
    )


def _restore_model(model: Any, snapshot: _ModelSnapshot) -> None:
    torch = _torch()
    state = model.state_dict(keep_vars=True)
    if set(state) != set(snapshot.tensor_values):
        raise Lfm2CpcfModelFoldInstallError(
            "Target state keys changed and cannot be transactionally restored."
        )
    observed_identity = {
        str(name): (
            id(value),
            int(value.data_ptr()),
            tuple(int(size) for size in value.shape),
            value.dtype,
        )
        for name, value in state.items()
    }
    if observed_identity != dict(snapshot.tensor_identity):
        raise Lfm2CpcfModelFoldInstallError(
            "Target tensor identities changed and cannot be restored safely."
        )
    with torch.no_grad():
        for name, value in state.items():
            value.copy_(
                snapshot.tensor_values[name].to(
                    device=value.device,
                    dtype=value.dtype,
                )
            )
    named_parameters = dict(model.named_parameters())
    if set(named_parameters) != set(snapshot.parameter_gradients):
        raise Lfm2CpcfModelFoldInstallError(
            "Target Parameter set changed during the model transaction."
        )
    for name, parameter in named_parameters.items():
        requires_grad, gradient, gradient_value = (
            snapshot.parameter_gradients[name]
        )
        parameter.requires_grad_(requires_grad)
        parameter.grad = gradient
        if gradient is not None:
            with torch.no_grad():
                gradient.copy_(
                    gradient_value.to(
                        device=gradient.device,
                        dtype=gradient.dtype,
                    )
                )
    modules = dict(model.named_modules())
    if set(modules) != {name for name, _mode in snapshot.module_modes}:
        raise Lfm2CpcfModelFoldInstallError(
            "Target module tree changed during the model transaction."
        )
    for name, mode in snapshot.module_modes:
        modules[name].training = mode
    if fingerprint_model_state(model) != snapshot.state_fingerprint:
        raise Lfm2CpcfModelFoldInstallError(
            "Target values failed exact whole-model rollback."
        )
    restored_state = model.state_dict(keep_vars=True)
    for name, expected in snapshot.tensor_identity.items():
        value = restored_state[name]
        observed = (
            id(value),
            int(value.data_ptr()),
            tuple(int(size) for size in value.shape),
            value.dtype,
        )
        if observed != expected:
            raise Lfm2CpcfModelFoldInstallError(
                "Target tensor identity changed during rollback."
            )
    for name, (requires_grad, gradient, gradient_value) in (
        snapshot.parameter_gradients.items()
    ):
        parameter = named_parameters[name]
        if (
            parameter.requires_grad != requires_grad
            or parameter.grad is not gradient
            or (
                gradient is not None
                and _tensor_sha256(gradient)
                != _tensor_sha256(gradient_value)
            )
        ):
            raise Lfm2CpcfModelFoldInstallError(
                "Target gradient state failed exact rollback."
            )
    if tuple(
        (name, bool(module.training))
        for name, module in model.named_modules()
    ) != snapshot.module_modes:
        raise Lfm2CpcfModelFoldInstallError(
            "Target training modes failed exact rollback."
        )


def _preserved_qk_norm_weights(
    source_model: Any,
    target_model: Any,
) -> dict[int, tuple[Any, Any]]:
    source_layers = tuple(_model_core(source_model, "source_model").layers)
    target_layers = tuple(_model_core(target_model, "target_model").layers)
    if len(source_layers) != len(target_layers):
        raise Lfm2CpcfModelFoldError(
            "Current LFM2 CPCF compiler requires equal source/target depth."
        )
    result = {}
    for index, (source, target) in enumerate(
        zip(source_layers, target_layers)
    ):
        if not bool(getattr(target, "is_attention_layer", False)):
            continue
        source_attention = getattr(source, "self_attn", None)
        target_attention = getattr(target, "self_attn", None)
        values = (
            getattr(
                getattr(source_attention, "q_layernorm", None),
                "weight",
                None,
            ),
            getattr(
                getattr(source_attention, "k_layernorm", None),
                "weight",
                None,
            ),
            getattr(
                getattr(target_attention, "q_layernorm", None),
                "weight",
                None,
            ),
            getattr(
                getattr(target_attention, "k_layernorm", None),
                "weight",
                None,
            ),
        )
        if any(value is None for value in values):
            raise Lfm2CpcfModelFoldError(
                f"Attention layer {index} lacks native Q/K RMSNorm gains."
            )
        source_q, source_k, target_q, target_k = values
        if (
            tuple(source_q.shape) == tuple(target_q.shape)
            and tuple(source_k.shape) == tuple(target_k.shape)
        ):
            result[index] = (
                source_q.detach().to(device="cpu"),
                source_k.detach().to(device="cpu"),
            )
    return result


def _final_norm_epsilon(target_model: Any) -> float:
    norm = getattr(
        _model_core(target_model, "target_model"),
        "embedding_norm",
        None,
    )
    for name in ("variance_epsilon", "eps", "epsilon"):
        value = getattr(norm, name, None)
        if value is None:
            continue
        try:
            epsilon = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(epsilon) and epsilon > 0.0:
            return epsilon
    raise Lfm2CpcfModelFoldError(
        "Target final RMSNorm exposes no positive epsilon."
    )


def _balanced_phase_weights(
    phases: Sequence[Lfm2CpcfTargetPhase],
    split: str,
) -> Any:
    torch = _torch()
    selected = tuple(phase for phase in phases if phase.split == split)
    if not selected:
        raise Lfm2CpcfModelFoldError(
            f"No {split!r} phases are available."
        )
    phase_mass = 1.0 / len(selected)
    pieces = []
    for phase in selected:
        rows = phase.batch_size * phase.sampled_suffix_tokens
        pieces.append(
            torch.full(
                (rows,),
                phase_mass / rows,
                dtype=torch.float64,
            )
        )
    return torch.cat(pieces).contiguous()


def _relative_rmse(prediction: Any, target: Any, weights: Any) -> float:
    numerator = (
        weights[:, None] * (prediction - target).square()
    ).sum()
    denominator = (weights[:, None] * target.square()).sum().clamp_min(1e-30)
    return float((numerator / denominator).sqrt().item())


def _retained_energy(prediction: Any, target: Any, weights: Any) -> float:
    error = (weights[:, None] * (prediction - target).square()).sum()
    energy = (weights[:, None] * target.square()).sum().clamp_min(1e-30)
    return float(max(0.0, 1.0 - float((error / energy).item())))


def _fit_and_install_final_norm(
    target_model: Any,
    *,
    phases: Sequence[Lfm2CpcfTargetPhase],
    final_state_bank: Lfm2CpcfPhaseStateBank,
    boundary_bundle: Lfm2CpcfBoundaryBundle,
    ridge: float,
) -> Mapping[str, Any]:
    torch = _torch()
    interface = boundary_bundle.interface
    split_rows: dict[str, tuple[Any, Any, Any]] = {}
    for split in ("fit", "validation"):
        selected = tuple(phase for phase in phases if phase.split == split)
        child = torch.cat(
            [
                final_state_bank.suffix(phase).reshape(
                    -1,
                    final_state_bank.hidden_size,
                )
                for phase in selected
            ],
            dim=0,
        ).to(device="cpu", dtype=torch.float64)
        parent = torch.cat(
            [
                phase.parent_evidence.final_norm_tensors[
                    "output.final_norm_output"
                ].reshape(-1, interface.source_width)
                for phase in selected
            ],
            dim=0,
        )
        target = interface.encode_producer_rows(parent)
        split_rows[split] = (
            child,
            target,
            _balanced_phase_weights(phases, split),
        )
    fit_child, fit_target, fit_weights = split_rows["fit"]
    fitted = fit_rmsnorm_diagonal(
        fit_child,
        fit_target,
        epsilon=_final_norm_epsilon(target_model),
        sample_weights=fit_weights,
        ridge=ridge,
    )
    parameter = getattr(
        getattr(
            _model_core(target_model, "target_model"),
            "embedding_norm",
            None,
        ),
        "weight",
        None,
    )
    if not isinstance(parameter, torch.nn.Parameter) or tuple(
        parameter.shape
    ) != tuple(fitted.weight.shape):
        raise Lfm2CpcfModelFoldError(
            "Target final RMSNorm differs from the fitted target gain."
        )
    identity = (
        id(parameter),
        int(parameter.data_ptr()),
        tuple(parameter.shape),
    )
    with torch.no_grad():
        parameter.copy_(
            fitted.weight.to(device=parameter.device, dtype=parameter.dtype)
        )
    if identity != (
        id(parameter),
        int(parameter.data_ptr()),
        tuple(parameter.shape),
    ):
        raise Lfm2CpcfModelFoldError(
            "Final RMSNorm installation replaced a target Parameter."
        )
    metrics = {}
    for split, (child, target, weights) in split_rows.items():
        prediction = fitted.predict(child)
        metrics[split] = {
            "rows": int(child.shape[0]),
            "relative_rmse": _relative_rmse(
                prediction,
                target,
                weights,
            ),
            "retained_energy_fraction": _retained_energy(
                prediction,
                target,
                weights,
            ),
        }
    body = {
        "component": "target_coordinate_final_rmsnorm",
        "fit_report_sha256": fitted.report["report_sha256"],
        "fit": metrics["fit"],
        "validation": metrics["validation"],
        "fit_validation_disjoint": True,
        "source_gain_projection_used": False,
        "parameter_identity_preserved": True,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
    }
    return MappingProxyType({**body, "report_sha256": _sha(body)})


@dataclass(frozen=True, slots=True)
class Lfm2CpcfModelFoldPolicy:
    """Whole-model coordinate, gate, and promotion controls."""

    ridge: float = 1e-8
    state_reconstruction_weight: float = 0.05
    minimum_predictive_energy_fraction: float = 0.90
    minimum_validation_block_retention: float = 0.0
    enforce_validation_block_floor: bool = True
    short_conv_gate_policy: Lfm2CpcfShortConvGatePolicy | None = None
    promotion_compression_ratio: float = 1.0
    require_promotion_ratio: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ridge",
            _nonnegative_float(self.ridge, "ridge"),
        )
        object.__setattr__(
            self,
            "state_reconstruction_weight",
            _nonnegative_float(
                self.state_reconstruction_weight,
                "state_reconstruction_weight",
            ),
        )
        object.__setattr__(
            self,
            "minimum_predictive_energy_fraction",
            _fraction(
                self.minimum_predictive_energy_fraction,
                "minimum_predictive_energy_fraction",
            ),
        )
        object.__setattr__(
            self,
            "minimum_validation_block_retention",
            _fraction(
                self.minimum_validation_block_retention,
                "minimum_validation_block_retention",
            ),
        )
        if not isinstance(self.enforce_validation_block_floor, bool):
            raise Lfm2CpcfModelFoldError(
                "enforce_validation_block_floor must be boolean."
            )
        if self.short_conv_gate_policy is not None and not isinstance(
            self.short_conv_gate_policy,
            Lfm2CpcfShortConvGatePolicy,
        ):
            raise TypeError(
                "short_conv_gate_policy must be "
                "Lfm2CpcfShortConvGatePolicy."
            )
        promotion = _nonnegative_float(
            self.promotion_compression_ratio,
            "promotion_compression_ratio",
        )
        if promotion <= 0.0:
            raise Lfm2CpcfModelFoldError(
                "promotion_compression_ratio must be positive."
            )
        object.__setattr__(
            self,
            "promotion_compression_ratio",
            promotion,
        )
        if not isinstance(self.require_promotion_ratio, bool):
            raise Lfm2CpcfModelFoldError(
                "require_promotion_ratio must be boolean."
            )

    def resolved_short_conv_policy(self) -> Lfm2CpcfShortConvGatePolicy:
        if self.short_conv_gate_policy is not None:
            return self.short_conv_gate_policy
        return Lfm2CpcfShortConvGatePolicy()

    def as_dict(self) -> dict[str, Any]:
        return {
            "ridge": self.ridge,
            "state_reconstruction_weight": (
                self.state_reconstruction_weight
            ),
            "minimum_predictive_energy_fraction": (
                self.minimum_predictive_energy_fraction
            ),
            "minimum_validation_block_retention": (
                self.minimum_validation_block_retention
            ),
            "enforce_validation_block_floor": (
                self.enforce_validation_block_floor
            ),
            "short_conv_gate_policy": (
                self.resolved_short_conv_policy().as_dict()
            ),
            "promotion_compression_ratio": (
                self.promotion_compression_ratio
            ),
            "require_promotion_ratio": self.require_promotion_ratio,
        }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfCompiledModelFold:
    """One completely installed and sequentially replayed LFM2 child."""

    boundary_bundle: Lfm2CpcfBoundaryBundle
    layer_results: tuple[Lfm2CpcfCompiledLayerResult, ...]
    final_state_bank: Lfm2CpcfPhaseStateBank
    parameter_receipt: Mapping[str, Any]
    interface_install_report: Mapping[str, Any]
    neutral_norm_install_report: Mapping[str, Any]
    final_norm_report: Mapping[str, Any]
    report: Mapping[str, Any]
    result_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.boundary_bundle, Lfm2CpcfBoundaryBundle):
            raise TypeError("boundary_bundle has the wrong type.")
        layers = tuple(self.layer_results)
        if not layers or tuple(
            result.layer_index for result in layers
        ) != tuple(range(len(layers))):
            raise Lfm2CpcfModelFoldError(
                "layer_results are incomplete or unordered."
            )
        if (
            not isinstance(self.final_state_bank, Lfm2CpcfPhaseStateBank)
            or self.final_state_bank.layer_index != len(layers)
        ):
            raise Lfm2CpcfModelFoldError(
                "Final state bank does not follow every target layer."
            )
        result_sha = _digest(self.result_sha256, "result_sha256")
        report = dict(self.report)
        if (
            report.pop("result_sha256", None) != result_sha
            or _sha(report) != result_sha
        ):
            raise Lfm2CpcfModelFoldError(
                "Compiled model-fold report seal is invalid."
            )
        object.__setattr__(self, "layer_results", layers)
        for name in (
            "parameter_receipt",
            "interface_install_report",
            "neutral_norm_install_report",
            "final_norm_report",
        ):
            object.__setattr__(
                self,
                name,
                MappingProxyType(dict(getattr(self, name))),
            )
        object.__setattr__(
            self,
            "report",
            MappingProxyType({**report, "result_sha256": result_sha}),
        )

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def _compile_lfm2_cpcf_model_fold_mutating(
    source_model: Any,
    target_model: Any,
    *,
    phases: Sequence[Lfm2CpcfTargetPhase],
    source_tied_table: Any,
    blank_target_report: Mapping[str, Any],
    planned_target_unique_parameters: int,
    policy: Lfm2CpcfModelFoldPolicy | None = None,
) -> Lfm2CpcfCompiledModelFold:
    """Mutating implementation guarded by the public whole-model transaction."""

    resolved_policy = policy or Lfm2CpcfModelFoldPolicy()
    if not isinstance(resolved_policy, Lfm2CpcfModelFoldPolicy):
        raise TypeError("policy must be Lfm2CpcfModelFoldPolicy.")
    ordered_phases = tuple(phases)
    if not ordered_phases or any(
        not isinstance(phase, Lfm2CpcfTargetPhase)
        for phase in ordered_phases
    ):
        raise Lfm2CpcfModelFoldError(
            "phases must contain aligned Lfm2CpcfTargetPhase evidence."
        )
    target_inventory_before = independent_parameter_inventory(target_model)
    target_count = int(target_inventory_before["unique_parameters"])
    planned = _positive_int(
        planned_target_unique_parameters,
        "planned_target_unique_parameters",
    )
    if target_count != planned:
        raise Lfm2CpcfModelFoldError(
            "Live target count differs from the planned dense blank."
        )
    blank = _blank_receipt(
        blank_target_report,
        target_unique_parameters=target_count,
    )
    _assert_target_is_zero_blank(target_model)
    source_layers = tuple(_model_core(source_model, "source_model").layers)
    target_layers = tuple(_model_core(target_model, "target_model").layers)
    if not source_layers or len(source_layers) != len(target_layers):
        raise Lfm2CpcfModelFoldError(
            "Source and target must expose equal nonzero LFM2 depth."
        )
    if len(ordered_phases[0].parent_evidence.layers) != len(source_layers):
        raise Lfm2CpcfModelFoldError(
            "Cached phase depth differs from the live source model."
        )
    source_state_before = fingerprint_model_state(source_model)
    parameter_receipt = audit_lfm2_cpcf_parameter_compaction(
        source_model,
        target_model,
        planned_target_unique_parameters=planned,
        promotion_compression_ratio=(
            resolved_policy.promotion_compression_ratio
        ),
    )
    if (
        resolved_policy.require_promotion_ratio
        and not parameter_receipt["promotion_ratio_met"]
    ):
        raise Lfm2CpcfModelFoldError(
            "Target is a valid diagnostic fold but does not meet the caller's "
            "promotion compression ratio."
        )
    boundary_bundle = fit_lfm2_cpcf_boundary_bundle(
        ordered_phases,
        source_tied_table,
        target_hidden_width=int(target_model.config.hidden_size),
        target_norm_epsilon=_final_norm_epsilon(target_model),
        state_reconstruction_weight=(
            resolved_policy.state_reconstruction_weight
        ),
        ridge=resolved_policy.ridge,
        minimum_predictive_energy_fraction=(
            resolved_policy.minimum_predictive_energy_fraction
        ),
        minimum_validation_block_retention=(
            resolved_policy.minimum_validation_block_retention
        ),
        enforce_validation_block_floor=(
            resolved_policy.enforce_validation_block_floor
        ),
    )
    interface_install = install_lfm2_cpcf_tied_interface(
        target_model,
        boundary_bundle.interface,
    )
    neutral_norm_install = install_lfm2_cpcf_neutral_norms(
        target_model,
        attention_qk_norm_weights=_preserved_qk_norm_weights(
            source_model,
            target_model,
        ),
    )
    state_bank = seed_lfm2_cpcf_phase_states(
        target_model,
        ordered_phases,
    )
    layer_results = []
    short_conv_policy = resolved_policy.resolved_short_conv_policy()
    for index, charts in enumerate(boundary_bundle.layers):
        if index != state_bank.layer_index:
            raise Lfm2CpcfModelFoldInstallError(
                "Sequential target replay attempted to skip a layer."
            )
        result = compile_and_install_lfm2_cpcf_layer(
            source_model,
            target_model,
            layer_index=index,
            input_bank=state_bank,
            boundary_charts=charts,
            ridge=resolved_policy.ridge,
            short_conv_gate_policy=short_conv_policy,
        )
        if result.layer_index != index:
            raise Lfm2CpcfModelFoldInstallError(
                "Layer compiler returned an out-of-order result."
            )
        layer_results.append(result)
        state_bank = result.next_bank
    if len(layer_results) != len(target_layers):
        raise Lfm2CpcfModelFoldInstallError(
            "Not every target macrocell was compiled and replayed."
        )
    final_norm_report = _fit_and_install_final_norm(
        target_model,
        phases=ordered_phases,
        final_state_bank=state_bank,
        boundary_bundle=boundary_bundle,
        ridge=resolved_policy.ridge,
    )
    target_inventory_after = independent_parameter_inventory(target_model)
    if (
        int(target_inventory_after["unique_parameters"]) != target_count
        or fingerprint_model_state(source_model) != source_state_before
    ):
        raise Lfm2CpcfModelFoldInstallError(
            "Model topology changed or source state was mutated during folding."
        )
    body = {
        "schema": LFM2_CPCF_MODEL_FOLD_SCHEMA,
        "blank_target_report_sha256": blank["report_sha256"],
        "policy": resolved_policy.as_dict(),
        "phase_count": len(ordered_phases),
        "phase_sha256": [phase.phase_sha256 for phase in ordered_phases],
        "boundary_bundle_sha256": boundary_bundle.bundle_sha256,
        "interface_install_report_sha256": interface_install[
            "report_sha256"
        ],
        "neutral_norm_install_report_sha256": neutral_norm_install[
            "report_sha256"
        ],
        "ordered_layer_result_sha256": [
            result.result_sha256 for result in layer_results
        ],
        "final_state_bank_sha256": state_bank.state_bank_sha256,
        "final_norm_report_sha256": final_norm_report["report_sha256"],
        "parameter_receipt": dict(parameter_receipt),
        "target_parameter_inventory_before": dict(
            target_inventory_before
        ),
        "target_parameter_inventory_after": dict(
            target_inventory_after
        ),
        "source_state_preserved": True,
        "target_parameter_count_preserved": True,
        "all_layers_compiled_in_order": True,
        "actual_folded_upstream_replay_used": True,
        "operation_audit": {
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
        },
    }
    result_sha = _sha(body)
    return Lfm2CpcfCompiledModelFold(
        boundary_bundle=boundary_bundle,
        layer_results=tuple(layer_results),
        final_state_bank=state_bank,
        parameter_receipt=parameter_receipt,
        interface_install_report=interface_install,
        neutral_norm_install_report=neutral_norm_install,
        final_norm_report=final_norm_report,
        report={**body, "result_sha256": result_sha},
        result_sha256=result_sha,
    )


def compile_lfm2_cpcf_model_fold(
    source_model: Any,
    target_model: Any,
    *,
    phases: Sequence[Lfm2CpcfTargetPhase],
    source_tied_table: Any,
    blank_target_report: Mapping[str, Any],
    planned_target_unique_parameters: int,
    policy: Lfm2CpcfModelFoldPolicy | None = None,
) -> Lfm2CpcfCompiledModelFold:
    """Compile every target layer as one rollback-safe model transaction."""

    source_before = fingerprint_model_state(source_model)
    target_before = _snapshot_model(target_model)
    try:
        return _compile_lfm2_cpcf_model_fold_mutating(
            source_model,
            target_model,
            phases=phases,
            source_tied_table=source_tied_table,
            blank_target_report=blank_target_report,
            planned_target_unique_parameters=(
                planned_target_unique_parameters
            ),
            policy=policy,
        )
    except Exception:
        try:
            _restore_model(target_model, target_before)
            if fingerprint_model_state(source_model) != source_before:
                raise Lfm2CpcfModelFoldInstallError(
                    "Source state changed during a failed model fold."
                )
        except Exception as rollback_exc:
            raise Lfm2CpcfModelFoldInstallError(
                "Whole-model CPCF compilation failed and exact target "
                "rollback could not be proven."
            ) from rollback_exc
        raise


__all__ = [
    "LFM2_CPCF_MODEL_FOLD_SCHEMA",
    "LFM2_CPCF_PARAMETER_RECEIPT_SCHEMA",
    "Lfm2CpcfCompiledModelFold",
    "Lfm2CpcfModelFoldError",
    "Lfm2CpcfModelFoldInstallError",
    "Lfm2CpcfModelFoldPolicy",
    "audit_lfm2_cpcf_parameter_compaction",
    "compile_lfm2_cpcf_model_fold",
    "independent_parameter_inventory",
]
