"""Exact, function-preserving expansion for supported HF causal decoders.

This module deliberately performs no optimization, backward pass, calibration,
or random selection.  It supports only architectures accepted by
``hf_arch_layout.build_decoder_layout`` and fails closed when its parameter
accounting assumptions do not hold.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from neural_forge.hf_arch_layout import (
    DecoderBlockLayout,
    DecoderLayout,
    MLPLayout,
    build_decoder_layout,
    resolve_module,
)


class HFExpansionError(ValueError):
    """Base error for fail-closed expansion."""


class NoExactExpansionPlanError(HFExpansionError):
    """Raised when supported operations cannot exactly reach the target."""


class ExpansionTopologyError(HFExpansionError):
    """Raised when the model violates an expansion proof assumption."""


@dataclass(frozen=True, slots=True)
class ExpansionOperation:
    kind: Literal["mlp_width", "identity_depth"]
    source_value: int
    target_value: int
    parameter_delta: int


@dataclass(frozen=True, slots=True)
class HFExpansionPlan:
    family: str
    source_parameter_count: int
    target_parameter_count: int
    fixed_parameter_count: int
    source_layers: int
    target_layers: int
    source_intermediate_size: int
    target_intermediate_size: int
    block_parameters_at_source_width: int
    parameters_per_block_channel: int
    added_layers: int
    added_channels: int
    donor_indices: tuple[int, ...]
    operations: tuple[ExpansionOperation, ...]
    selection_policy: str = "min_added_units_then_depth_then_width"

    @property
    def parameter_delta(self) -> int:
        return self.target_parameter_count - self.source_parameter_count


@dataclass(frozen=True, slots=True)
class SourceCoverage:
    source_layer_count: int
    depth_donor_indices: tuple[int, ...]
    depth_donor_coverage: float
    source_channel_count: int
    replicated_source_channels: tuple[int, ...]
    width_source_coverage: float


@dataclass(frozen=True, slots=True)
class InsertedLayerProvenance:
    inserted_index: int
    donor_index: int
    donor_layer_type: str | None
    zeroed_operator_paths: tuple[str, str]
    zeroed_parameter_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HFExpansionReport:
    plan: HFExpansionPlan
    actual_source_parameter_count: int
    actual_target_parameter_count: int
    actual_parameter_delta: int
    width_parameter_delta: int
    depth_parameter_delta: int
    function_equivalence_max_error: float | None
    validation_tensor_count: int
    source_coverage: SourceCoverage
    inserted_layer_provenance: tuple[InsertedLayerProvenance, ...]
    topology_validated: bool


@dataclass(frozen=True, slots=True)
class HFExpansionResult:
    model: Any
    report: HFExpansionReport


@dataclass(frozen=True, slots=True)
class _Accounting:
    layout: DecoderLayout
    total: int
    block_parameters: int
    fixed_parameters: int
    channel_parameters: int


def count_unique_parameters(model: Any) -> int:
    """Count parameter elements once, including when weights are tied."""

    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        identity = id(parameter)
        if identity not in seen:
            seen.add(identity)
            total += int(parameter.numel())
    return total


def _direct_parameter_names(module: Any, label: str) -> set[str]:
    names = {name for name, _ in module.named_parameters(recurse=False)}
    if names - {"weight", "bias"} or "weight" not in names:
        raise ExpansionTopologyError(
            f"{label} must expose only a weight and optional bias parameter."
        )
    return names


def _projection_channel_cost(
    module: Any,
    axis: int,
    width: int,
    label: str,
    *,
    bias_expands: bool,
) -> int:
    names = _direct_parameter_names(module, label)
    weight = module.weight
    if getattr(weight, "ndim", None) != 2 or int(weight.shape[axis]) != width:
        raise ExpansionTopologyError(
            f"{label} does not expose the validated intermediate axis {axis}."
        )
    cost = int(weight.numel()) // width
    bias = getattr(module, "bias", None)
    if "bias" in names:
        if bias is None or getattr(bias, "ndim", None) != 1:
            raise ExpansionTopologyError(f"{label} has malformed bias state.")
        if bias_expands and int(bias.numel()) == width:
            cost += 1
        elif bias_expands:
            raise ExpansionTopologyError(
                f"{label} bias does not follow its expanded channel axis."
            )
    return cost


def _mlp_channel_cost(model: Any, mlp: MLPLayout, family: str) -> int:
    width = mlp.intermediate_size
    if mlp.projection_weight_layout == "output_input":
        incoming_axis, outgoing_axis = 0, 1
    elif mlp.projection_weight_layout == "input_output":
        incoming_axis, outgoing_axis = 1, 0
    else:  # pragma: no cover - guarded by DecoderLayout's literal contract
        raise ExpansionTopologyError("Unknown MLP projection orientation.")

    incoming_paths = (
        (mlp.gate_projection_path, mlp.up_projection_path)
        if mlp.kind == "gated"
        else (mlp.input_projection_path,)
    )
    if any(path is None for path in incoming_paths):
        raise ExpansionTopologyError("MLP layout is missing an incoming projection.")
    cost = 0
    for path in incoming_paths:
        assert path is not None
        module = resolve_module(model, path, family=family)
        cost += _projection_channel_cost(
            module, incoming_axis, width, path, bias_expands=True
        )
    down = resolve_module(model, mlp.down_projection_path, family=family)
    cost += _projection_channel_cost(
        down,
        outgoing_axis,
        width,
        mlp.down_projection_path,
        bias_expands=False,
    )
    return cost


def _model_accounting(model: Any) -> _Accounting:
    layout = build_decoder_layout(model)
    blocks = resolve_module(model, layout.blocks_path, family=layout.family)
    if len(blocks) != layout.layer_count or not blocks:
        raise ExpansionTopologyError("Decoder block container is inconsistent.")

    block_parameter_ids: list[set[int]] = []
    block_counts: list[int] = []
    channel_costs: list[int] = []
    for index, block in enumerate(blocks):
        named_parameters = list(
            block.named_parameters(recurse=True, remove_duplicate=False)
        )
        identities = {id(parameter) for _, parameter in named_parameters}
        if len(identities) != len(named_parameters):
            raise ExpansionTopologyError(
                f"Decoder block {index} contains shared parameter objects."
            )
        block_parameter_ids.append(identities)
        block_counts.append(
            sum(int(parameter.numel()) for _, parameter in named_parameters)
        )
        channel_costs.append(
            _mlp_channel_cost(model, layout.blocks[index].mlp, layout.family)
        )

    all_block_ids: set[int] = set()
    for index, identities in enumerate(block_parameter_ids):
        overlap = all_block_ids & identities
        if overlap:
            raise ExpansionTopologyError(
                f"Decoder block {index} shares parameters with another block."
            )
        all_block_ids.update(identities)
    occurrences: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(
        recurse=True, remove_duplicate=False
    ):
        occurrences.setdefault(id(parameter), []).append(name)
    block_prefix = f"{layout.blocks_path}."
    for identity in all_block_ids:
        names = occurrences.get(identity, [])
        if any(not name.startswith(block_prefix) for name in names):
            raise ExpansionTopologyError(
                "A decoder block parameter is shared outside the block stack."
            )
    if len(set(block_counts)) != 1:
        raise ExpansionTopologyError(
            "Exact planning requires equal unique parameter counts in every decoder block."
        )
    if len(set(channel_costs)) != 1:
        raise ExpansionTopologyError(
            "Exact planning requires equal MLP channel costs in every decoder block."
        )

    total = count_unique_parameters(model)
    block_parameters = block_counts[0]
    fixed = total - layout.layer_count * block_parameters
    if fixed < 0:
        raise ExpansionTopologyError("Unique parameter decomposition is inconsistent.")
    return _Accounting(
        layout=layout,
        total=total,
        block_parameters=block_parameters,
        fixed_parameters=fixed,
        channel_parameters=channel_costs[0],
    )


def _positive_divisors(value: int) -> tuple[int, ...]:
    low: list[int] = []
    high: list[int] = []
    for divisor in range(1, math.isqrt(value) + 1):
        if value % divisor == 0:
            low.append(divisor)
            quotient = value // divisor
            if quotient != divisor:
                high.append(quotient)
    return tuple(low + list(reversed(high)))


def plan_hf_expansion(model: Any, target_parameter_count: int) -> HFExpansionPlan:
    """Plan an exact deterministic depth/MLP-width expansion without mutation."""

    if isinstance(target_parameter_count, bool) or not isinstance(
        target_parameter_count, int
    ):
        raise TypeError("target_parameter_count must be an integer.")
    accounting = _model_accounting(model)
    if target_parameter_count < accounting.total:
        raise HFExpansionError(
            f"Target {target_parameter_count} is below source count {accounting.total}."
        )

    layout = accounting.layout
    residual_block_budget = target_parameter_count - accounting.fixed_parameters
    if residual_block_budget <= 0:
        raise NoExactExpansionPlanError(
            f"Target {target_parameter_count} cannot contain the fixed parameter set."
        )

    candidates: list[tuple[int, int]] = []
    for target_layers in _positive_divisors(residual_block_budget):
        if target_layers < layout.layer_count:
            continue
        final_block_parameters = residual_block_budget // target_layers
        extra_per_block = final_block_parameters - accounting.block_parameters
        if extra_per_block < 0 or extra_per_block % accounting.channel_parameters:
            continue
        added_channels = extra_per_block // accounting.channel_parameters
        candidates.append((target_layers - layout.layer_count, added_channels))
    if not candidates:
        raise NoExactExpansionPlanError(
            "Exact target is unreachable by uniform Net2Wider MLP channels and "
            f"identity blocks (source={accounting.total}, target={target_parameter_count}, "
            f"block={accounting.block_parameters}, channel_per_block="
            f"{accounting.channel_parameters})."
        )

    added_layers, added_channels = min(
        candidates, key=lambda item: (item[0] + item[1], item[0], item[1])
    )
    target_layers = layout.layer_count + added_layers
    target_width = layout.intermediate_size + added_channels
    donor_indices = tuple(index % layout.layer_count for index in range(added_layers))
    width_delta = layout.layer_count * added_channels * accounting.channel_parameters
    widened_block_parameters = (
        accounting.block_parameters + added_channels * accounting.channel_parameters
    )
    depth_delta = added_layers * widened_block_parameters
    operations: list[ExpansionOperation] = []
    if added_channels:
        operations.append(
            ExpansionOperation(
                kind="mlp_width",
                source_value=layout.intermediate_size,
                target_value=target_width,
                parameter_delta=width_delta,
            )
        )
    if added_layers:
        operations.append(
            ExpansionOperation(
                kind="identity_depth",
                source_value=layout.layer_count,
                target_value=target_layers,
                parameter_delta=depth_delta,
            )
        )
    if accounting.total + width_delta + depth_delta != target_parameter_count:
        raise ExpansionTopologyError("Internal exact-budget proof failed.")
    return HFExpansionPlan(
        family=layout.family,
        source_parameter_count=accounting.total,
        target_parameter_count=target_parameter_count,
        fixed_parameter_count=accounting.fixed_parameters,
        source_layers=layout.layer_count,
        target_layers=target_layers,
        source_intermediate_size=layout.intermediate_size,
        target_intermediate_size=target_width,
        block_parameters_at_source_width=accounting.block_parameters,
        parameters_per_block_channel=accounting.channel_parameters,
        added_layers=added_layers,
        added_channels=added_channels,
        donor_indices=donor_indices,
        operations=tuple(operations),
    )


def _new_parameter(torch: Any, source: Any, value: Any) -> Any:
    return torch.nn.Parameter(
        value.to(device=source.device, dtype=source.dtype),
        requires_grad=source.requires_grad,
    )


def _expanded_indices(torch: Any, old_width: int, added: int, device: Any) -> Any:
    extras = torch.arange(added, device=device, dtype=torch.long).remainder(old_width)
    return torch.cat(
        (torch.arange(old_width, device=device, dtype=torch.long), extras), dim=0
    )


def _replica_counts(torch: Any, indices: Any, old_width: int, dtype: Any) -> Any:
    return torch.bincount(indices, minlength=old_width).to(dtype=dtype)


def _widen_incoming_projection(
    torch: Any,
    module: Any,
    *,
    axis: int,
    indices: Any,
    old_width: int,
    label: str,
) -> None:
    _direct_parameter_names(module, label)
    old_weight = module.weight
    local_indices = indices.to(old_weight.device)
    new_weight = old_weight.detach().index_select(axis, local_indices).clone()
    module.weight = _new_parameter(torch, old_weight, new_weight)
    bias = getattr(module, "bias", None)
    if bias is not None:
        if int(bias.numel()) != old_width:
            raise ExpansionTopologyError(f"{label} bias does not match output width.")
        module.bias = _new_parameter(
            torch,
            bias,
            bias.detach().index_select(0, local_indices.to(bias.device)).clone(),
        )
    if hasattr(module, "out_features") and axis == 0:
        module.out_features = int(indices.numel())
    if hasattr(module, "in_features") and axis == 1:
        module.in_features = int(indices.numel())
    if hasattr(module, "nf") and axis == 1:
        module.nf = int(indices.numel())


def _widen_outgoing_projection(
    torch: Any,
    module: Any,
    *,
    axis: int,
    indices: Any,
    counts: Any,
    label: str,
) -> None:
    _direct_parameter_names(module, label)
    old_weight = module.weight
    local_indices = indices.to(old_weight.device)
    local_counts = counts.to(old_weight.device)
    divisors = local_counts.index_select(0, local_indices).to(
        device=old_weight.device, dtype=old_weight.dtype
    )
    shape = [1, 1]
    shape[axis] = int(divisors.numel())
    new_weight = old_weight.detach().index_select(axis, local_indices).clone()
    new_weight = new_weight / divisors.reshape(shape)
    module.weight = _new_parameter(torch, old_weight, new_weight)
    if hasattr(module, "in_features") and axis == 1:
        module.in_features = int(indices.numel())
    if hasattr(module, "out_features") and axis == 0:
        module.out_features = int(indices.numel())


def _update_intermediate_metadata(config: Any, family: str, value: int) -> None:
    targets = [config]
    text_config = getattr(config, "text_config", None)
    if text_config is not None and text_config is not config:
        targets.append(text_config)
    field = "n_inner" if family == "gpt2" else "intermediate_size"
    updated = False
    for target in targets:
        if hasattr(target, field):
            setattr(target, field, int(value))
            updated = True
    if not updated:
        raise ExpansionTopologyError(f"Configuration has no {field!r} width field.")


def _widen_all_mlps(model: Any, layout: DecoderLayout, added: int) -> tuple[int, ...]:
    if added == 0:
        return ()
    import torch

    old_width = layout.intermediate_size
    source_channels = tuple(index % old_width for index in range(added))
    for block in layout.blocks:
        mlp = block.mlp
        if mlp.projection_weight_layout == "output_input":
            incoming_axis, outgoing_axis = 0, 1
        else:
            incoming_axis, outgoing_axis = 1, 0
        incoming_paths = (
            (mlp.gate_projection_path, mlp.up_projection_path)
            if mlp.kind == "gated"
            else (mlp.input_projection_path,)
        )
        first_path = incoming_paths[0]
        assert first_path is not None
        first = resolve_module(model, first_path, family=layout.family)
        indices = _expanded_indices(torch, old_width, added, first.weight.device)
        counts = _replica_counts(torch, indices, old_width, first.weight.dtype)
        for path in incoming_paths:
            assert path is not None
            module = resolve_module(model, path, family=layout.family)
            _widen_incoming_projection(
                torch,
                module,
                axis=incoming_axis,
                indices=indices,
                old_width=old_width,
                label=path,
            )
        down = resolve_module(model, mlp.down_projection_path, family=layout.family)
        _widen_outgoing_projection(
            torch,
            down,
            axis=outgoing_axis,
            indices=indices.to(down.weight.device),
            counts=counts.to(down.weight.device),
            label=mlp.down_projection_path,
        )
    _update_intermediate_metadata(
        model.config, layout.family, old_width + added
    )
    return source_channels


def _relative_path(block: DecoderBlockLayout, operator_path: str) -> str:
    prefix = f"{block.path}."
    if not operator_path.startswith(prefix):
        raise ExpansionTopologyError(
            f"Residual output {operator_path!r} is outside donor {block.path!r}."
        )
    return operator_path[len(prefix) :]


def _resolve_relative(module: Any, path: str) -> Any:
    current = module
    for segment in path.split("."):
        current = current[int(segment)] if segment.isdecimal() else getattr(current, segment)
    return current


def _zero_residual_outputs(
    donor: Any, block: DecoderBlockLayout, inserted_index: int
) -> tuple[tuple[str, str], tuple[str, ...]]:
    import torch

    if block.residual.kind not in {"serial_pre_norm", "serial_pre_post_norm"}:
        raise ExpansionTopologyError(
            f"Residual topology {block.residual.kind!r} has no identity proof."
        )
    relative_paths = (
        _relative_path(block, block.attention.output_path),
        _relative_path(block, block.mlp.down_projection_path),
    )
    target_paths = tuple(
        f"{block.path.rsplit('.', 1)[0]}.{inserted_index}.{path}"
        for path in relative_paths
    )
    zeroed: list[str] = []
    with torch.no_grad():
        for target_path, relative_path in zip(target_paths, relative_paths):
            operator = _resolve_relative(donor, relative_path)
            names = _direct_parameter_names(operator, target_path)
            for name in ("weight", "bias"):
                if name in names:
                    parameter = getattr(operator, name)
                    parameter.zero_()
                    zeroed.append(f"{target_path}.{name}")
    return (target_paths[0], target_paths[1]), tuple(zeroed)


def _config_targets(config: Any) -> tuple[Any, ...]:
    text_config = getattr(config, "text_config", None)
    if text_config is not None and text_config is not config:
        return config, text_config
    return (config,)


def _append_layer_metadata(
    config: Any, old_count: int, donor_indices: tuple[int, ...]
) -> None:
    new_count = old_count + len(donor_indices)
    count_updated = False
    for target in _config_targets(config):
        for field in ("num_hidden_layers", "n_layer", "num_layers"):
            if hasattr(target, field):
                try:
                    setattr(target, field, int(new_count))
                    count_updated = True
                except (AttributeError, TypeError):
                    continue
        layer_types = getattr(target, "layer_types", None)
        if layer_types is not None:
            if not isinstance(layer_types, (list, tuple)) or len(layer_types) != old_count:
                raise ExpansionTopologyError(
                    "Configuration layer_types does not match source depth."
                )
            source = list(layer_types)
            target.layer_types = source + [source[index] for index in donor_indices]
    if not count_updated:
        raise ExpansionTopologyError("Configuration has no writable layer-count field.")


def _set_layer_indices(block: Any, index: int) -> None:
    for module in block.modules():
        if hasattr(module, "layer_idx"):
            module.layer_idx = int(index)


def _append_identity_blocks(
    model: Any, layout: DecoderLayout, donor_indices: tuple[int, ...]
) -> tuple[InsertedLayerProvenance, ...]:
    if not donor_indices:
        return ()
    blocks = resolve_module(model, layout.blocks_path, family=layout.family)
    original_blocks = tuple(blocks)
    provenance: list[InsertedLayerProvenance] = []
    for offset, donor_index in enumerate(donor_indices):
        inserted_index = layout.layer_count + offset
        donor_layout = layout.blocks[donor_index]
        donor = copy.deepcopy(original_blocks[donor_index])
        zeroed_paths, zeroed_parameters = _zero_residual_outputs(
            donor, donor_layout, inserted_index
        )
        _set_layer_indices(donor, inserted_index)
        blocks.append(donor)
        provenance.append(
            InsertedLayerProvenance(
                inserted_index=inserted_index,
                donor_index=donor_index,
                donor_layer_type=donor_layout.attention.layer_type,
                zeroed_operator_paths=zeroed_paths,
                zeroed_parameter_names=zeroed_parameters,
            )
        )
    _append_layer_metadata(model.config, layout.layer_count, donor_indices)
    return tuple(provenance)


def _training_snapshot(model: Any) -> tuple[tuple[Any, bool], ...]:
    return tuple((module, bool(module.training)) for module in model.modules())


def _restore_training(snapshot: tuple[tuple[Any, bool], ...]) -> None:
    for module, training in snapshot:
        module.training = training


def _extract_logits(output: Any) -> Any:
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, Mapping) and "logits" in output:
        return output["logits"]
    raise HFExpansionError("Validation forward output does not expose logits.")


def _validation_error(
    source: Any, candidate: Any, validation_inputs: Mapping[str, Any]
) -> tuple[float, int]:
    import torch

    source_snapshot = _training_snapshot(source)
    candidate_snapshot = _training_snapshot(candidate)
    source.eval()
    candidate.eval()
    try:
        with torch.inference_mode():
            before = _extract_logits(source(**dict(validation_inputs))).detach()
            after = _extract_logits(candidate(**dict(validation_inputs))).detach()
        if tuple(before.shape) != tuple(after.shape):
            raise HFExpansionError(
                f"Validation logits changed shape from {tuple(before.shape)} to "
                f"{tuple(after.shape)}."
            )
        if not bool(torch.isfinite(before).all()) or not bool(torch.isfinite(after).all()):
            raise HFExpansionError("Validation logits must be finite.")
        error = float(
            (before.to(torch.float64) - after.to(torch.float64))
            .abs()
            .max()
            .cpu()
        )
        return error, 1
    finally:
        _restore_training(source_snapshot)
        _restore_training(candidate_snapshot)


def expand_hf_model(
    model: Any,
    target_parameter_count: int,
    *,
    validation_inputs: Mapping[str, Any] | None = None,
    max_equivalence_error: float | None = None,
) -> HFExpansionResult:
    """Return a function-preserving expanded copy at an exact parameter count.

    The caller's model is never mutated.  Validation, when supplied, is an
    inference-only before/after logits comparison.  ``max_equivalence_error``
    optionally turns that measurement into a fail-closed acceptance gate.
    """

    if max_equivalence_error is not None and (
        not math.isfinite(max_equivalence_error) or max_equivalence_error < 0
    ):
        raise ValueError("max_equivalence_error must be finite and nonnegative.")
    if max_equivalence_error is not None and validation_inputs is None:
        raise ValueError("max_equivalence_error requires validation_inputs.")

    plan = plan_hf_expansion(model, target_parameter_count)
    candidate = copy.deepcopy(model)
    copied_source_count = count_unique_parameters(candidate)
    if copied_source_count != plan.source_parameter_count:
        raise ExpansionTopologyError(
            "Deep copy did not preserve the source's unique parameter count."
        )

    layout = build_decoder_layout(candidate)
    before_width = copied_source_count
    replicated_channels = _widen_all_mlps(candidate, layout, plan.added_channels)
    width_count = count_unique_parameters(candidate)
    if width_count - before_width != sum(
        op.parameter_delta for op in plan.operations if op.kind == "mlp_width"
    ):
        raise ExpansionTopologyError("Observed Net2Wider parameter delta disagrees with plan.")
    widened_layout = build_decoder_layout(candidate)

    before_depth = width_count
    provenance = _append_identity_blocks(
        candidate, widened_layout, plan.donor_indices
    )
    final_count = count_unique_parameters(candidate)
    if final_count - before_depth != sum(
        op.parameter_delta for op in plan.operations if op.kind == "identity_depth"
    ):
        raise ExpansionTopologyError("Observed identity-depth parameter delta disagrees with plan.")
    final_layout = build_decoder_layout(candidate)
    if (
        final_count != plan.target_parameter_count
        or final_layout.layer_count != plan.target_layers
        or final_layout.intermediate_size != plan.target_intermediate_size
    ):
        raise ExpansionTopologyError("Materialized model disagrees with the exact plan.")

    max_error = None
    validation_tensor_count = 0
    if validation_inputs is not None:
        max_error, validation_tensor_count = _validation_error(
            model, candidate, validation_inputs
        )
        if max_equivalence_error is not None and max_error > max_equivalence_error:
            raise HFExpansionError(
                f"Function-equivalence error {max_error:.9g} exceeds "
                f"limit {max_equivalence_error:.9g}."
            )

    depth_sources = tuple(sorted(set(plan.donor_indices)))
    width_sources = tuple(sorted(set(replicated_channels)))
    coverage = SourceCoverage(
        source_layer_count=plan.source_layers,
        depth_donor_indices=depth_sources,
        depth_donor_coverage=(
            len(depth_sources) / plan.source_layers if plan.added_layers else 0.0
        ),
        source_channel_count=plan.source_intermediate_size,
        replicated_source_channels=width_sources,
        width_source_coverage=(
            len(width_sources) / plan.source_intermediate_size
            if plan.added_channels
            else 0.0
        ),
    )
    report = HFExpansionReport(
        plan=plan,
        actual_source_parameter_count=plan.source_parameter_count,
        actual_target_parameter_count=final_count,
        actual_parameter_delta=final_count - plan.source_parameter_count,
        width_parameter_delta=width_count - before_width,
        depth_parameter_delta=final_count - before_depth,
        function_equivalence_max_error=max_error,
        validation_tensor_count=validation_tensor_count,
        source_coverage=coverage,
        inserted_layer_provenance=provenance,
        topology_validated=True,
    )
    return HFExpansionResult(model=candidate, report=report)


def expand_model_to_budget(
    model: Any,
    target_parameter_count: int,
    *,
    validation_inputs: Mapping[str, Any] | None = None,
    max_equivalence_error: float | None = None,
) -> HFExpansionResult:
    """Descriptive alias for :func:`expand_hf_model`."""

    return expand_hf_model(
        model,
        target_parameter_count,
        validation_inputs=validation_inputs,
        max_equivalence_error=max_equivalence_error,
    )


__all__ = [
    "ExpansionOperation",
    "ExpansionTopologyError",
    "HFExpansionError",
    "HFExpansionPlan",
    "HFExpansionReport",
    "HFExpansionResult",
    "InsertedLayerProvenance",
    "NoExactExpansionPlanError",
    "SourceCoverage",
    "count_unique_parameters",
    "expand_hf_model",
    "expand_model_to_budget",
    "plan_hf_expansion",
]
